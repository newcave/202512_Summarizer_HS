import os
import time
import tempfile
from typing import Optional, Tuple

import streamlit as st
from google import genai
from pypdf import PdfReader


# ============================================================
# 0) App Config
# ============================================================
st.set_page_config(
    page_title="KIHS 보고서 분석기 (데모)",
    page_icon="💧",
    layout="wide",
)

st.title("💧 KIHS (한국수자원조사기술원) 보고서 분석기 (Demo)")
st.caption("PDF 요약 · 정책/기술 시사점 · 객관식 퀴즈 생성 — Gemini + 로컬 파싱 우선(안정)")

st.markdown(
    """
- 본 앱은 **데모**입니다. 결과는 참고용이며, 원문 근거 범위 내에서만 해석해야 합니다.
- **API 안정성**을 위해 먼저 PDF를 로컬에서 텍스트로 추출한 뒤, 텍스트 기반으로 Gemini에 질의합니다.
- 스캔 PDF 등 텍스트 추출이 부족한 경우에만(선택) 파일 업로드 방식으로 **fallback**합니다.
"""
)

# ============================================================
# 1) Prompt Definitions (사전 정의)
#    - (2) 출력 섹션 정의
#    - (3) 옵션 정의 (톤/언어/금지/형식/근거)
# ============================================================

# (A) 공통 규칙/옵션: "항상 적용"
PROMPT_COMMON_RULES = """
[공통 규칙]
- 반드시 한국어로 답변하세요.
- 당신은 'KIHS(한국수자원조사기술원) 보고서 분석가'입니다.
- 문서에 없는 내용은 만들지 말고, 불확실하면 '문서에서 확인 불가'라고 명시하세요.
- 가능한 경우, 근거가 되는 문서 표현을 짧게 요약하여 함께 제시하세요(직접 인용은 1문장 이내).
- 과장 없이 간결하고 단정한 문장으로 작성하세요.
"""

# (B) 출력 섹션(2,3 등) 고정: 요약 리포트
PROMPT_SECTIONS_SUMMARY = """
[출력 섹션]
1) 핵심 요약 (6줄 이내)
2) 연구 배경/문제정의 (bullet 3~6개)
3) 주요 성과/결과 (bullet 5~10개, 가능하면 정량/수치 포함)
4) 결론 (bullet 3~6개)
5) 정책 시사점 (3~6개, 실행형 문장)
6) 기술 시사점 (3~6개, 실행형 문장)
7) 한계/리스크/전제 (bullet 3~8개)
8) 다음 단계 제안 (3~6개)
"""

# (C) 퀴즈 출력 포맷: 객관식
PROMPT_SECTIONS_QUIZ = """
[출력 형식(퀴즈)]
- 문항 수: {num_q}문항
- 각 문항은 다음 형식으로만 작성:

Q1. (문제)
A) 보기
B) 보기
C) 보기
D) 보기
정답: (A/B/C/D)
해설: (문서 근거 기반 2~4줄)

- 모든 문항은 문서 내용에 근거해야 하며, 추측/창작 금지.
"""

# (D) 옵션(3): 스타일/톤/레벨
PROMPT_OPTIONS = """
[옵션]
- 톤: 공공기관 보고서 스타일(차분, 단정, 과장 없음)
- 독자: 수자원/물관리 분야 실무자 및 연구자
- 금지: 홍보성 표현, 선정적/감정적 표현, 근거 없는 단정
- 용어: 가능하면 한국어 용어 우선(예: water treatment plant=정수장)
"""

# (E) 개별 Task 프롬프트 템플릿
TASK_SUMMARY = """
[작업]
업로드된 KIHS 보고서(PDF)의 내용을 바탕으로, 아래 섹션에 맞춰 요약 보고서를 작성하세요.
"""

TASK_QUIZ = """
[작업]
업로드된 KIHS 보고서(PDF)의 내용을 바탕으로, 핵심 이해도를 점검하는 객관식 퀴즈를 생성하세요.
"""

# ============================================================
# 2) API Key + Client
# ============================================================
def get_api_key() -> Optional[str]:
    key = (st.secrets.get("GOOGLE_API_KEY") or "").strip()
    if key:
        return key
    # 데모용 수기 입력(운영 배포는 Secrets 권장)
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
# 6) Build Final Prompt (사전정의 결합)
# ============================================================
def build_prompt_for_summary(doc_text: str) -> str:
    return f"""
{PROMPT_COMMON_RULES}
{PROMPT_OPTIONS}
{PROMPT_SECTIONS_SUMMARY}

{TASK_SUMMARY}

[문서 텍스트]
{doc_text}
""".strip()


def build_prompt_for_quiz(doc_text: str, num_q: int) -> str:
    quiz_format = PROMPT_SECTIONS_QUIZ.format(num_q=num_q)
    return f"""
{PROMPT_COMMON_RULES}
{PROMPT_OPTIONS}
{quiz_format}

{TASK_QUIZ}

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

# 1) 로컬 파싱(우선)
MIN_TEXT_CHARS = 1200  # 이보다 작으면 텍스트 기반 분석이 부정확/불가할 수 있음
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
tab1, tab2, tab3 = st.tabs(["📄 요약 리포트", "🎓 객관식 퀴즈", "🧾 파싱 확인"])

with tab3:
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
    st.caption("기본은 텍스트 기반 분석(안정). 텍스트가 부족하면 파일 기반 분석(대체)을 사용합니다.")

    btn_summary = st.button("요약 리포트 생성", type="primary", key="btn_summary")
    if btn_summary:
        with st.spinner("리포트 생성 중..."):
            try:
                # 텍스트 기반 우선
                if st.session_state.parsed_text and len(st.session_state.parsed_text) >= MIN_TEXT_CHARS:
                    doc_text = trim_text(st.session_state.parsed_text, max_chars=max_chars)
                    prompt = build_prompt_for_summary(doc_text)
                    out = generate_with_retry(model=model, contents=prompt, retries=1)
                    st.markdown(out)

                # 업로드 대체 경로
                elif st.session_state.file_ref is not None:
                    prompt = (
                        f"{PROMPT_COMMON_RULES}\n{PROMPT_OPTIONS}\n{PROMPT_SECTIONS_SUMMARY}\n\n"
                        f"{TASK_SUMMARY}\n\n"
                        "※ 문서 텍스트 추출이 부족하여 파일 기반으로 분석합니다."
                    )
                    out = generate_with_retry(model=model, contents=[st.session_state.file_ref, prompt], retries=1)
                    st.markdown(out)

                else:
                    st.error("텍스트도 부족하고 업로드 대체 경로도 준비되지 않았습니다. (옵션/네트워크 확인)")

            except Exception as e:
                st.error("요약 리포트 생성 중 오류가 발생했습니다.")
                st.exception(e)

# ------------------------------------------------------------
# Tab2: Quiz
# ------------------------------------------------------------
with tab2:
    st.markdown("### 🧠 객관식 퀴즈 생성")
    st.caption("문서 내용 기반으로 이해도 점검용 문항을 생성합니다.")
    num_q = st.slider("문항 수", 1, 8, 3, key="num_q")

    btn_quiz = st.button("퀴즈 생성", type="secondary", key="btn_quiz")
    if btn_quiz:
        with st.spinner("퀴즈 생성 중..."):
            try:
                # 텍스트 기반 우선
                if st.session_state.parsed_text and len(st.session_state.parsed_text) >= MIN_TEXT_CHARS:
                    doc_text = trim_text(st.session_state.parsed_text, max_chars=max_chars)
                    prompt = build_prompt_for_quiz(doc_text, num_q=num_q)
                    out = generate_with_retry(model=model, contents=prompt, retries=1)
                    st.markdown(out)

                # 업로드 대체 경로
                elif st.session_state.file_ref is not None:
                    prompt = (
                        f"{PROMPT_COMMON_RULES}\n{PROMPT_OPTIONS}\n"
                        f"{PROMPT_SECTIONS_QUIZ.format(num_q=num_q)}\n\n"
                        f"{TASK_QUIZ}\n\n"
                        "※ 문서 텍스트 추출이 부족하여 파일 기반으로 분석합니다."
                    )
                    out = generate_with_retry(model=model, contents=[st.session_state.file_ref, prompt], retries=1)
                    st.markdown(out)

                else:
                    st.error("텍스트도 부족하고 업로드 대체 경로도 준비되지 않았습니다. (옵션/네트워크 확인)")

            except Exception as e:
                st.error("퀴즈 생성 중 오류가 발생했습니다.")
                st.exception(e)
