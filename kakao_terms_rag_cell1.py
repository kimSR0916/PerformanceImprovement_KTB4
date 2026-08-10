# -*- coding: utf-8 -*-
# =====================================================================================
# [학생용] 결과기 구현 — 1번 셀 (팀 구현)
# RAG 구성: 카카오 공식 정책 페이지 실시간 수집 → 조(條) 단위 청크 →
#           KURE-v1(nlpai-lab) 밀집 임베딩 + BM25 희소 검색 하이브리드 →
#           Qwen2.5-7B-Instruct 4bit(NF4) 생성
# =====================================================================================

# -------------------------------------------------------------------------------------
# 0. 패키지 설치 (torch는 Colab 기본 설치본을 그대로 사용, 재설치하지 않음)
# -------------------------------------------------------------------------------------
import subprocess
import sys


def _pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


_pip_install(
    [
        "transformers>=4.46.0",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers",
        "faiss-cpu",
        "rank_bm25",
        "beautifulsoup4",
        "requests",
    ]
)

import re
import unicodedata

import faiss
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OFFICIAL_DOCUMENT_NAMES = (
    "카카오계정 약관",
    "카카오 위치정보 이용약관",
    "카카오 통합서비스약관",
    "카카오 통합 약관",
)

# -------------------------------------------------------------------------------------
# 1. 약관 원문 수집 — 카카오 공식 정책 페이지를 실시간으로 가져와 파싱한다
#    (Drive 마운트/사전 업로드 금지 조건을 지키기 위해 매 실행마다 새로 수집)
# -------------------------------------------------------------------------------------
# 후보 URL은 여러 개를 등록해 두고 위에서부터 순서대로 시도한다.
# 카카오가 페이지 구조를 바꾸면 이 표만 갱신하면 된다.
TERMS_SOURCES = {
    "카카오계정 약관": [
        "https://www.kakao.com/policy/terms?type=a&lang=ko",
    ],
    "카카오 위치정보 이용약관": [
        "https://www.kakao.com/policy/location?lang=ko",
    ],
    "카카오 통합서비스약관": [
        "https://www.kakao.com/policy/terms?type=ts&lang=ko",
    ],
    "카카오 통합 약관": [
        "https://www.kakao.com/policy/terms?type=t&lang=ko",
        # 확인된 별도 페이지가 없을 경우를 대비한 폴백(통합서비스약관과 동일 소스).
        "https://www.kakao.com/policy/terms?type=ts&lang=ko",
    ],
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; KTB-RAG-Bot/1.0)"}

# 줄 맨 앞의 "제N조(제목)" 패턴만 조 경계로 인정한다.
# (본문 중간에 "제3조에 따라..."처럼 조문을 인용하는 문장과 구분하기 위함)
_ARTICLE_PATTERN = re.compile(r"^제\s*(\d+)\s*조\s*(?:\(([^)]{0,80})\))?", re.MULTILINE)


def _fetch_html(url: str):
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[fetch 실패] {url} -> {e}")
        return None


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = unicodedata.normalize("NFKC", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _split_articles(text: str):
    """본문을 '제N조' 단위로 분리해 (조번호, 조 전체텍스트) 리스트를 반환."""
    matches = list(_ARTICLE_PATTERN.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_no = int(m.group(1))
        body = text[start:end].strip()
        if len(body) < 5:
            continue
        chunks.append((article_no, body))
    return chunks


def _fetch_terms_document(doc_name: str):
    for url in TERMS_SOURCES[doc_name]:
        html = _fetch_html(url)
        if not html:
            continue
        text = _extract_visible_text(html)
        articles = _split_articles(text)
        if articles:
            print(f"[수집 완료] {doc_name}: {len(articles)}개 조 (source={url})")
            return articles
        print(f"[조 추출 실패, 다음 후보 시도] {doc_name} <- {url}")
    print(f"[경고] {doc_name} 원문을 확보하지 못했습니다. 이 문서는 검색에서 제외됩니다.")
    return []


def _subsplit_long_article(body: str, max_chars: int = 700):
    """조 하나가 너무 길면 문단 단위로 서브 청크로 나눈다. 조 번호는 상위에서 그대로 유지."""
    if len(body) <= max_chars:
        return [body]
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


print("[1/5] 약관 원문 수집 중...")
_RAW_CORPUS = []  # [{"doc": str, "article_no": int, "text": str}, ...]
for _doc_name in OFFICIAL_DOCUMENT_NAMES:
    for _article_no, _body in _fetch_terms_document(_doc_name):
        for _sub in _subsplit_long_article(_body):
            _RAW_CORPUS.append({"doc": _doc_name, "article_no": _article_no, "text": _sub})

if not _RAW_CORPUS:
    raise RuntimeError("약관 원문을 하나도 확보하지 못했습니다. TERMS_SOURCES의 URL을 점검하세요.")

print(f"[1/5] 총 {len(_RAW_CORPUS)}개 청크 확보")

# -------------------------------------------------------------------------------------
# 2. 밀집 검색 — KURE-v1 임베딩 + FAISS
# -------------------------------------------------------------------------------------
print("[2/5] KURE 임베딩 모델 로드 중...")
_EMBED_MODEL = SentenceTransformer(
    "nlpai-lab/KURE-v1", device="cuda" if torch.cuda.is_available() else "cpu"
)

_CORPUS_TEXTS = [c["text"] for c in _RAW_CORPUS]
_CORPUS_EMB = _EMBED_MODEL.encode(
    _CORPUS_TEXTS,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype("float32")

_FAISS_INDEX = faiss.IndexFlatIP(_CORPUS_EMB.shape[1])
_FAISS_INDEX.add(_CORPUS_EMB)
print(f"[2/5] FAISS 인덱스 구축 완료 ({_FAISS_INDEX.ntotal}개 벡터)")

# -------------------------------------------------------------------------------------
# 3. 희소 검색 — BM25 (법률/약관 특유의 고유명사·숫자 매칭 보완용)
# -------------------------------------------------------------------------------------
print("[3/5] BM25 희소 검색 인덱스 구축 중...")
_TOKEN_PATTERN = re.compile(r"[가-힣]+|[a-zA-Z0-9]+")


def _tokenize(text: str):
    return _TOKEN_PATTERN.findall(text.lower())


_BM25 = BM25Okapi([_tokenize(t) for t in _CORPUS_TEXTS])


def _min_max_norm(arr):
    arr = np.asarray(arr, dtype="float32")
    span = arr.max() - arr.min()
    if span < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span


def _hybrid_search(query: str, top_k: int = 4, dense_weight: float = 0.65):
    q_emb = _EMBED_MODEL.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    pool = min(20, len(_RAW_CORPUS))
    dense_scores, dense_idx = _FAISS_INDEX.search(q_emb, pool)
    dense_scores, dense_idx = dense_scores[0], dense_idx[0]

    bm25_scores = _BM25.get_scores(_tokenize(query))
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:pool]

    candidate_idx = sorted(set(dense_idx.tolist()) | set(bm25_top_idx.tolist()))
    dense_map = {i: s for i, s in zip(dense_idx.tolist(), dense_scores.tolist())}

    dense_vals = _min_max_norm([dense_map.get(i, 0.0) for i in candidate_idx])
    bm25_vals = _min_max_norm([bm25_scores[i] for i in candidate_idx])
    combined = dense_weight * dense_vals + (1 - dense_weight) * bm25_vals

    ranked = sorted(zip(candidate_idx, combined), key=lambda x: x[1], reverse=True)

    results, seen = [], set()
    for idx, score in ranked:
        item = _RAW_CORPUS[idx]
        key = (item["doc"], item["article_no"])
        if key in seen:
            continue
        seen.add(key)
        results.append({**item, "score": float(score)})
        if len(results) >= top_k:
            break
    return results


# -------------------------------------------------------------------------------------
# 4. 생성 모델 — Qwen2.5-7B-Instruct, 4bit(NF4) 양자화
# -------------------------------------------------------------------------------------
print("[4/5] Qwen2.5-7B-Instruct(4bit) 로드 중...")
_GEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

_TOKENIZER = AutoTokenizer.from_pretrained(_GEN_MODEL_ID)
_GEN_MODEL = AutoModelForCausalLM.from_pretrained(
    _GEN_MODEL_ID,
    quantization_config=_bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
_GEN_MODEL.eval()
print("[4/5] 생성 모델 로드 완료")

_SYSTEM_PROMPT = (
    "당신은 카카오 약관 안내 상담원입니다. 아래 [근거 조항]에 있는 내용만 근거로 "
    "질문에 정확하고 간결하게 답하세요.\n"
    "- 근거 조항에 없는 내용은 추측하지 말고 '약관에서 해당 내용을 찾을 수 없습니다.'라고 답하세요.\n"
    "- 답변 중에 근거가 된 문서명과 조 번호를 자연스럽게 언급하세요.\n"
    "- 불필요한 서론 없이 답변만 작성하세요."
)


def _build_prompt(question: str, contexts: list) -> str:
    context_block = "\n\n".join(
        f"[근거 {i + 1}] {c['doc']} 제{c['article_no']}조\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    user_content = f"[근거 조항]\n{context_block}\n\n[질문]\n{question}"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def _generate(prompt: str, max_new_tokens: int = 400) -> str:
    inputs = _TOKENIZER(prompt, return_tensors="pt").to(_GEN_MODEL.device)
    output_ids = _GEN_MODEL.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.05,
        pad_token_id=_TOKENIZER.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


# =====================================================================================
# 1-1. 팀별 자유 구현 영역 — 고정 진입점
# =====================================================================================
def answer_question(question: str):
    """공통 러너가 질문마다 호출하는 고정 진입점입니다.

    RAG 파이프라인: 하이브리드 검색(KURE 밀집 + BM25 희소)으로 관련 조항을 추린 뒤,
    해당 조항만 프롬프트에 넣어 Qwen2.5-7B-Instruct(4bit)로 답변을 생성한다.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question은 비어 있지 않은 문자열이어야 합니다.")

    question = question.strip()
    contexts = _hybrid_search(question, top_k=4)

    if not contexts:
        return {"answer": "약관에서 관련 근거를 찾을 수 없습니다.", "retrieved": []}

    prompt = _build_prompt(question, contexts)
    answer_text = _generate(prompt)
    retrieved = [[c["doc"], c["article_no"]] for c in contexts]
    return {"answer": answer_text, "retrieved": retrieved}


print("[5/5] 결과기 준비 완료. 예시 질의로 파이프라인을 점검합니다...")
try:
    _sample = answer_question("카카오계정을 탈퇴하려면 어떻게 해야 하나요?")
    print("예시 답변:", _sample["answer"][:200])
    print("예시 근거:", _sample["retrieved"])
except Exception as e:
    print(f"[경고] 예시 질의 실행 중 오류: {e}")


# =====================================================================================
# 2. 고정 FastAPI 연결 영역 — 삭제하거나 경로를 바꾸지 않습니다 (원본 그대로)
# =====================================================================================
import threading  # noqa: E402


def _install_server_packages():
    """공통 러너와 연결하는 데 필요한 가벼운 서버 패키지만 설치합니다."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn"],
        check=True,
    )


_install_server_packages()

from fastapi import FastAPI, HTTPException  # noqa: E402

app = FastAPI(title="KTB AI Performance Result Generator")
_GENERATION_LOCK = threading.Lock()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/answer")
def answer_api(payload: dict):
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="question must be a non-empty string")
    with _GENERATION_LOCK:
        return answer_question(question.strip())


print("[1번 셀 준비] 결과기 구현을 마친 뒤 2번 공통 러너를 실행하세요.")
