import streamlit as st
import os
import tempfile
from google import genai
from google.genai import types

# --- 페이지 설정 ---
st.set_page_config(
    page_title="KIHS 수자원 데이터 분석기",
    page_icon="💧",
    layout="wide"
)

# --- 헤더 섹션 ---
st.title("💧 한국수자원조사기술원(KIHS) AI 분석기")
st.subheader("수자원 포럼 및 보고서 PDF 분석 (Demo)")
st.markdown("""
이 대시보드는 **Google Gemini 1.5 Flash** 모델을 활용하여 
KIHS 보고서(PDF)를 요약하고 핵심 내용을 바탕으로 퀴즈를 생성합니다.
""")

# --- 사이드바: 설정 및 파일 업로드 ---
with st.sidebar:
    st.header("설정 및 업로드")
    
    # API 키 처리 (st.secrets 또는 직접 입력)
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("Google API Key를 입력하세요", type="password")
    
    if not api_key:
        st.warning("앱을 사용하려면 API Key가 필요합니다.")
        st.stop()

    # 클라이언트 초기화
    client = genai.Client(api_key=api_key)

    # 파일 업로드
    uploaded_file = st.file_uploader("KIHS 보고서(PDF) 업로드", type=["pdf"])

# --- 메인 기능 함수 ---
def upload_to_gemini(uploaded_file):
    """스트림릿 업로드 파일을 로컬 임시파일로 저장 후 Gemini에 업로드"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        with st.spinner("Gemini 서버로 문서를 전송 중입니다..."):
            # Gemini File API를 통해 파일 업로드
            file_ref = client.files.upload(path=tmp_path)
            # 처리가 완료될 때까지 대기 (대용량 파일의 경우 필요할 수 있음)
            # 보통 텍스트 위주의 PDF는 즉시 처리됨
        return file_ref
    except Exception as e:
        st.error(f"파일 업로드 중 오류 발생: {e}")
        return None

# --- 메인 로직 ---
if uploaded_file:
    # 1. 파일 업로드 상태 관리 (세션 스테이트 활용)
    if "file_ref" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
        st.session_state.file_ref = upload_to_gemini(uploaded_file)
        st.session_state.last_uploaded = uploaded_file.name
        st.success(f"문서 업로드 완료! ({uploaded_file.name})")

    file_ref = st.session_state.file_ref

    if file_ref:
        # 탭을 사용하여 기능 분리
        tab1, tab2 = st.tabs(["📄 문서 요약", "🎓 핵심 퀴즈"])

        # --- 탭 1: 문서 요약 ---
        with tab1:
            st.markdown("### 📋 보고서 주요 내용 요약")
            if st.button("요약 생성하기", type="primary"):
                with st.spinner("AI가 문서를 분석하고 요약 중입니다..."):
                    try:
                        prompt = "이 수자원 관련 보고서의 핵심 내용을 요약해줘. 특히 연구의 배경, 주요 성과, 그리고 결론을 중심으로 정리해줘."
                        response = client.models.generate_content(
                            model="gemini-1.5-flash", # 안정성을 위해 1.5-flash 사용 (2.0 등 변경 가능)
                            contents=[file_ref, prompt]
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"요약 생성 중 오류가 발생했습니다: {e}")

        # --- 탭 2: 퀴즈 생성 ---
        with tab2:
            st.markdown("### 🧠 이해도 점검 퀴즈")
            st.info("보고서 내용을 바탕으로 객관식 퀴즈를 생성합니다.")
            
            num_quiz = st.slider("생성할 문제 수", 1, 5, 3)
            
            if st.button("퀴즈 만들기"):
                with st.spinner("AI가 퀴즈를 출제 중입니다..."):
                    try:
                        prompt = f"""
                        이 문서를 바탕으로 수자원 전문가를 위한 객관식 퀴즈 {num_quiz}문제를 만들어줘.
                        형식은 다음과 같이 해줘:
                        
                        1. 문제 내용
                        A) 보기1
                        B) 보기2
                        C) 보기3
                        D) 보기4
                        
                        [정답 및 해설]
                        정답: (번호)
                        해설: (이유)
                        
                        ---
                        """
                        response = client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=[file_ref, prompt]
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"퀴즈 생성 중 오류가 발생했습니다: {e}")

else:
    st.info("좌측 사이드바에서 PDF 파일을 업로드해주세요.")
    # 데모용 안내 이미지 또는 텍스트
    st.markdown("---")
    st.markdown("**사용 예시:**")
    st.markdown("- 2021_KIHS_수자원포럼_최종보고서.pdf")
    st.markdown("- 2022_KIHS_수자원포럼_최종보고서.pdf")
