import os
import time
import tempfile
from typing import Optional, Tuple, List, Dict

import streamlit as st
from google import genai
from pypdf import PdfReader


# ============================================================
# 0) App Config
# ============================================================
st.set_page_config(
    page_title="KIHS 보고서 학습형 분석기 (데모)",
    page_icon="💧",
    layout="wide",
)

st.title("💧 KIHS (한국수자원조사기술원) 보고서 학습형 분석기 (Demo)")
st.caption("PDF 요약 · 교육형 Q&A · 추가 질의(대화) — Gemini + 로컬 파싱 우선(안정)")

st.markdown(
    """
- 본 앱은 **데모**입니다. 결과는 참고용이며, 원문 근거 범위 내에서만 해석해야 합니다.
- **API 안정성**을 위해 먼저 PDF를 로컬에서 텍스트로 추출한 뒤, 텍스트 기반으로 Gemini에 질의합니다.
- 스캔 PDF 등 텍스트 추출이 부족한 경우에만(선택) 파일 업로드 방식으로 **fallback**합니다.
"""
)

# ============================================================
# 1) Prompt Definitions (사전 정의)
# ============================================================

PROMPT_COMMON_RULES = """
[공통 규칙]
- 반드시 한국어로 답변하세요.
- 당신은 'KIHS(한국수자원조사기술원) 보고서 기반 교육/분석 튜터'입니다.
- 문서에 없는 내용은 만들지 말고, 불확실하면 '문서에서 확인 불가'라고 명시하세요.
- 가능한 경우, 근거가 되는 문서 표현을 1~2문장으로 요약해 함께 제시하세요(직접 인용은 1문장 이내).
- 과장 없이 간결하고 단정한 문장으로 작성하세요.
"""

PROMPT_OPTIONS = """
[옵션]
- 톤: 공공기관 보고서 스타일(차분, 단정, 과장 없음) + 학습자 친화(핵심→설명→정리)
- 독자: 수자원/물관리 분야 실무자 및 연구자(초중급 포함)
- 금지: 홍보성 표현, 감정적 표현, 근거 없는 단정, 과도한 상상
- 용어: 한국어 용어 우선(예: water treatment plant=정수장)
"""

PROMPT_SECTIONS_SUMMARY = """
[요약 리포트 출력 섹션]
1) 핵심 요약 (6줄 이내)
2) 연구 배경/문제정의 (bullet 3~6개)
3) 방법/데이터/대상 (bullet 3~8개) — 문서에서 확인되는 범위
4) 주요 결과/성과 (bullet 5~12개, 가능하면 수치 포함)
5) 결론 (bullet 3~6개)
6) 정책 시사점 (3~6개, 실행형)
7) 기술 시사점 (3~6개, 실행형)
8) 한계/리스크/전제 (bullet 3~8개)
9) 다음 단계/추가 연구 질문 (3~6개)
"""

PROMPT_EDU_QA_SPEC = """
[교육형 Q&A 생성 규격]
- 총 {num_q}개 문항을 생성.
- 형식은 아래 고정:

Q1. (질문: 개념/맥락/근거 중심)
A1. (짧은 답: 3~5줄)
근거(문서 기반): (문서에서 확인되는 근거를 1~2문장으로 요약)
추가 설명: (배경 설명/오해 방지 3~6줄)
학습 체크: (예/아니오 또는 단답형 질문 1개)

- 질문 유형은 섞어서 구성:
  (a) 핵심 개념 정의  (b) 왜 중요한가(맥락)  (c) 방법/데이터  (d) 결과 해석  (e) 한계/리스크  (f) 실무 적용
"""

PROMPT_CHAT_SPEC = """
[추가 질의(대화) 규칙]
- 사용자의 질문에 대해, 문서 근거를 최우선으로 답변.
- 문서에 없는 내용은 '문서에서 확인 불가'로 처리하고, 대신 확인을 위한 질문/추가자료를 제안.
- 답변 구조:
  1) 결론(2~4줄)
  2) 근거(문서 기반 bullet)
  3) 실무/정책/기술 시사점(가능 시 bullet)
  4) 추가 확인 질문(1~3개)
"""

TASK_SUMMARY = """
[작업]
업로드된 KIHS 보고서(PDF)의 내용을 바탕으로, 지정된 섹션 형식에 맞춰 요약 리포트를 작성하세요.
"""

TASK_EDU_QA = """
[작업]
업로드된 KIHS 보고서(PDF)의 내용을 바탕으로, 학습용 교육형 Q&A를 생성하세요.
"""

TASK_CHAT = """
[작업]
아래 사용자의 추가 질문에 대해, 문서 근거 중심으로 답하세요.
"""

# ============================================================
# 2) API Key + Client
# ============================================================
def get_api_key() -> Optional[str]:
    key = (st.secrets.get("GOOGLE_API_KEY") or "").strip()
    if key:
        return key
    with st.sidebar:
        st.warning("Secrets에 GOOGLE_API_KEY가 없어 입력 모드로 전환되었습니다(데모용).")
        key2 = st.text_input("Google API Key 입력", type="password").strip()
        return key2 if key2 else None


api_key = get_api_key()
if not api_key:
    st.warning("API Key가 필요합니다. Streamlit Cloud → Secrets에 GOOGLE_API_KEY 설정을 권장합니다.")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("Gemini Client 초기화 실패")
    st.exception(e)
    st.stop()

# ============================================================
# 3) PDF Parsing (Primary path for stability)
# ============================================================
def extract_text_from_pdf(uploaded_file) -> Tuple[str, int]:
    reader = PdfReader(uploaded_file)
    n_pages = len(reader.pages)

    parts = []
    for i in range(n_pages):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if t:
            parts.append(f"[PAGE {i+1}]\n{t}")

    return "\n\n".join(parts).strip(), n_pages


def normalize_text(text: str) -> str:
    t = (text or "").replace("\r", "\n")
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t.strip()


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...입력 길이 제한으로 일부 생략됨...]"


# ============================================================
# 4) Gemini File Upload Fallback (optional)
# ============================================================
def upload_pdf_to_gemini_file_api(client, uploaded_file) -> Optional[object]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("Gemini 서버로 PDF 업로드 중(대체 경로)..."):
            file_ref = client.files.upload(path=tmp_path)

        return file_ref
    except Exception as e:
        st.error("파일 업로드 중 오류가 발생했습니다.")
        st.exception(e)
        return None
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ============================================================
# 5) Gemini Call Wrapper (minimal retry)
# ============================================================
def generate_with_retry(model: str, contents, retries: int = 1, sleep_s: float = 0.6) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            return resp.text or ""
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s)
    raise last_err


# ============================================================
# 6) Build Prompts
# ============================================================
def build_prompt_summary(doc_text: str) -> str:
    return f"""
{PROMPT_COMMON_RULES}
{PROMPT_OPTIONS}
{PROMPT_SECTIONS_SUMMARY}

{TASK_SUMMARY}

[문서 텍스트]
{doc_text}
""".strip()


def build_prompt_edu_qa(doc_text: str, num_q: int) -> str:
    spec = PROMPT_EDU_QA_SPEC.format(num_q=num_q)
    return f"""
{PROMPT_COMMON_RULES}
{PROMPT_OPTIONS}
{spec}

{TASK_EDU_QA}

[문서 텍스트]
{doc_text}
""".strip()


def build_prompt_chat(doc_text: str, chat_history: List[Dict[str, str]], user_q: str) -> str:
    # 히스토리는 길이 제한이 필요 (너무 길면 API 문제)
    # 최근 N턴만 포함
    last_turns = chat_history[-6:] if chat_history else []
    history_txt = "\n".join([f"{m['role']}: {m['content']}" for m in last_turns])

    return f"""
{PROMPT_COMMON_RULES}
{PROMPT_OPTIONS}
{PROMPT_CHAT_SPEC}

{TASK_CHAT}

[대화 기록(최근)]
{history_txt}

[사용자 질문]
{user_q}

[문서 텍스트]
{doc_text}
""".strip()


# ============================================================
# 7) Sidebar UI (Korean)
# ============================================================
with st.sidebar:
    st.header("설정 및 업로드")

    model = st.selectbox("모델 선택", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0)

    max_chars = st.slider("문서 입력 상한(문자 수)", 8000, 60000, 24000, 2000)

    st.subheader("처리 모드")
    prefer_local_parse = st.checkbox("로컬 텍스트 파싱 우선(권장)", value=True)
    allow_file_fallback = st.checkbox("텍스트 부족 시 업로드 대체 경로 허용", value=True)

    uploaded_file = st.file_uploader("KIHS 보고서 PDF 업로드", type=["pdf"])

    st.subheader("교육형 Q&A 설정")
    num_q = st.slider("문항 수", 3, 15, 7)

    st.markdown("### ⚠️ 안내")
    st.info(
        "- 스캔 PDF(이미지)는 텍스트 추출이 거의 안 될 수 있습니다.\n"
        "- 그 경우 업로드 대체 경로를 켜면 파일 기반 분석을 시도합니다.\n"
        "- 네트워크/정책에 따라 업로드는 실패할 수 있습니다."
    )


# ============================================================
# 8) Session State
# ============================================================
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None
if "parsed_text" not in st.session_state:
    st.session_state.parsed_text = ""
if "n_pages" not in st.session_state:
    st.session_state.n_pages = 0
if "file_ref" not in st.session_state:
    st.session_state.file_ref = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content}


# ============================================================
# 9) Main Logic
# ============================================================
if not uploaded_file:
    st.info("좌측 사이드바에서 PDF 파일을 업로드하세요.")
    st.markdown("---")
    st.markdown("**예시 파일:**")
    st.markdown("- 2021_KIHS_Water Resources Forum_Final Report.pdf")
    st.markdown("- 2022_KIHS_Water Resources Forum_Final Report.pdf")
    st.stop()

# 새 파일이면 상태 초기화
if st.session_state.last_uploaded != uploaded_file.name:
    st.session_state.last_uploaded = uploaded_file.name
    st.session_state.parsed_text = ""
    st.session_state.n_pages = 0
    st.session_state.file_ref = None
    st.session_state.chat_history = []

# 1) 로컬 파싱(우선)
MIN_TEXT_CHARS = 1200
if prefer_local_parse and not st.session_state.parsed_text:
    with st.spinner("PDF 텍스트 추출(로컬 파싱) 중..."):
        try:
            text, n_pages = extract_text_from_pdf(uploaded_file)
            text = normalize_text(text)
            st.session_state.parsed_text = text
            st.session_state.n_pages = n_pages
        except Exception as e:
            st.error("PDF 텍스트 추출 실패")
            st.exception(e)

st.success(f"문서 로드 완료: {uploaded_file.name}")
st.caption(f"페이지 수: {st.session_state.n_pages} | 추출 텍스트 길이: {len(st.session_state.parsed_text):,} chars")

# 2) 텍스트 부족하면 업로드 대체 경로(선택)
text_insufficient = len(st.session_state.parsed_text) < MIN_TEXT_CHARS
if text_insufficient and allow_file_fallback and st.session_state.file_ref is None:
    st.warning("텍스트 추출이 부족합니다(스캔 PDF 가능). 업로드 대체 경로를 시도합니다.")
    st.session_state.file_ref = upload_pdf_to_gemini_file_api(client, uploaded_file)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 요약 리포트", "🎓 교육형 Q&A", "💬 추가 질의", "🧾 파싱 확인"])

with tab4:
    st.markdown("### 🧾 텍스트 파싱 확인(일부)")
    if st.session_state.parsed_text:
        st.text_area("미리보기", trim_text(st.session_state.parsed_text, 4000), height=260)
    else:
        st.info("추출된 텍스트가 없습니다. (스캔 PDF일 가능성)")

# ------------------------------------------------------------
# Tab1: Summary Report
# ------------------------------------------------------------
with tab1:
    st.markdown("### 📋 요약 리포트 생성")
    btn_summary = st.button("요약 리포트 생성", type="primary", key="btn_summary")
    if btn_summary:
        with st.spinner("리포트 생성 중..."):
            try:
                if st.session_state.parsed_text and len(st.session_state.parsed_text) >= MIN_TEXT_CHARS:
                    doc_text = trim_text(st.session_state.parsed_text, max_chars=max_chars)
                    prompt = build_prompt_summary(doc_text)
                    out = generate_with_retry(model=model, contents=prompt, retries=1)
                    st.markdown(out)
                elif st.session_state.file_ref is not None:
                    prompt = (
                        f"{PROMPT_COMMON_RULES}\n{PROMPT_OPTIONS}\n{PROMPT_SECTIONS_SUMMARY}\n\n"
                        f"{TASK_SUMMARY}\n\n"
                        "※ 문서 텍스트 추출이 부족하여 파일 기반으로 분석합니다."
                    )
                    out = generate_with_retry(model=model, contents=[st.session_state.file_ref, prompt], retries=1)
                    st.markdown(out)
                else:
                    st.error("텍스트도 부족하고 업로드 대체 경로도 준비되지 않았습니다.")
            except Exception as e:
                st.error("요약 리포트 생성 중 오류가 발생했습니다.")
                st.exception(e)

# ------------------------------------------------------------
# Tab2: Educational Q&A
# ------------------------------------------------------------
with tab2:
    st.markdown("### 🎓 교육형 Q&A 생성")
    st.caption("문서 기반으로 핵심 개념/맥락/방법/결과/한계/실무 적용을 학습하도록 문답을 구성합니다.")

    btn_qa = st.button("교육형 Q&A 생성", type="secondary", key="btn_qa")
    if btn_qa:
        with st.spinner("교육형 Q&A 생성 중..."):
            try:
                if st.session_state.parsed_text and len(st.session_state.parsed_text) >= MIN_TEXT_CHARS:
                    doc_text = trim_text(st.session_state.parsed_text, max_chars=max_chars)
                    prompt = build_prompt_edu_qa(doc_text, num_q=num_q)
                    out = generate_with_retry(model=model, contents=prompt, retries=1)
                    st.markdown(out)
                elif st.session_state.file_ref is not None:
                    prompt = (
                        f"{PROMPT_COMMON_RULES}\n{PROMPT_OPTIONS}\n"
                        f"{PROMPT_EDU_QA_SPEC.format(num_q=num_q)}\n\n"
                        f"{TASK_EDU_QA}\n\n"
                        "※ 문서 텍스트 추출이 부족하여 파일 기반으로 분석합니다."
                    )
                    out = generate_with_retry(model=model, contents=[st.session_state.file_ref, prompt], retries=1)
                    st.markdown(out)
                else:
                    st.error("텍스트도 부족하고 업로드 대체 경로도 준비되지 않았습니다.")
            except Exception as e:
                st.error("교육형 Q&A 생성 중 오류가 발생했습니다.")
                st.exception(e)

# ------------------------------------------------------------
# Tab3: Chat / Follow-up queries
# ------------------------------------------------------------
with tab3:
    st.markdown("### 💬 추가 질의 (문서 기반 Q&A)")
    st.caption("보고서 내용에 대해 추가 질문을 입력하면, 문서 근거 중심으로 답합니다.")

    # 대화 표시
    if st.session_state.chat_history:
        for m in st.session_state.chat_history:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.markdown(m["content"])
    else:
        st.info("아직 대화가 없습니다. 아래 입력창에 질문을 입력해 보세요.")

    user_q = st.chat_input("추가 질문을 입력하세요 (예: 이 보고서의 핵심 데이터는 무엇인가요?)")

    if user_q:
        # store user msg
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.spinner("답변 생성 중..."):
            try:
                # 텍스트 기반 우선
                if st.session_state.parsed_text and len(st.session_state.parsed_text) >= MIN_TEXT_CHARS:
                    doc_text = trim_text(st.session_state.parsed_text, max_chars=max_chars)
                    prompt = build_prompt_chat(doc_text, st.session_state.chat_history, user_q)
                    out = generate_with_retry(model=model, contents=prompt, retries=1)

                # 업로드 대체 경로
                elif st.session_state.file_ref is not None:
                    prompt = (
                        f"{PROMPT_COMMON_RULES}\n{PROMPT_OPTIONS}\n{PROMPT_CHAT_SPEC}\n\n"
                        f"{TASK_CHAT}\n\n"
                        f"[사용자 질문]\n{user_q}\n\n"
                        "※ 문서 텍스트 추출이 부족하여 파일 기반으로 분석합니다."
                    )
                    out = generate_with_retry(model=model, contents=[st.session_state.file_ref, prompt], retries=1)
                else:
                    out = "텍스트도 부족하고 업로드 대체 경로도 준비되지 않았습니다. (옵션/네트워크 확인)"

                # store assistant msg
                st.session_state.chat_history.append({"role": "assistant", "content": out})

                with st.chat_message("assistant"):
                    st.markdown(out)

            except Exception as e:
                st.error("추가 질의 처리 중 오류가 발생했습니다.")
                st.exception(e)
