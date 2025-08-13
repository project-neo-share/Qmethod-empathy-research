"""
Q-Method (TADT) Streamlit Application — Safe Init & Required Fields

Author      : Your Team
Last Update : 2025-08-14
Description : Likert-based Q-Method survey tool for TADT (Tech-Affective Dynamics Theory)
              - 시나리오(고객센터/의료/교육) '필수' 선택
              - 이메일 '필수' 입력(간단 유효성 검증)
              - Likert (1~5) Q-set 응답 수집 (최대 30문항)
              - Factor Analysis(가능하면) 또는 PCA로 유형(Type) 도출
              - 전략 매트릭스 매핑:
                   (1) 예측(Predictive) ↔ 공감(Empathy)
                   (2) 위임(Delegation) ↔ 협업(Collaboration)
              - 유형별 운영모델/인사전략/서비스혁신 권고 자동 생성
              - 빈 파일/파일 없음/파싱에러 대비, 저장/분석 단계 예외 처리
"""

import os
import re
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# FactorAnalyzer optional
_FA_AVAILABLE = True
try:
    from factor_analyzer import FactorAnalyzer
except Exception:
    _FA_AVAILABLE = False
    from sklearn.decomposition import PCA

# -----------------------------
# Page & Globals
# -----------------------------
st.set_page_config(page_title="Q-Method (TADT) Analyzer", layout="wide")
st.title("Q-Method (TADT) Likert Analyzer — Safe & Required")

DATA_PATH = "responses_tadt.csv"
MIN_N_FOR_ANALYSIS = 5  # 최소 유효 응답 수

# -----------------------------
# Korean font (optional)
# -----------------------------
try:
    import matplotlib.font_manager as fm
    font_path = "fonts/NanumGothic.ttf"
    if os.path.exists(font_path):
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
except Exception:
    pass

# -----------------------------
# Utilities
# -----------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def is_valid_email(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) > 150:
        return False
    return bool(EMAIL_RE.match(s))

def load_csv_safe(path: str):
    """Return DataFrame or None if file not found/empty/parse error."""
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    except Exception as e:
        st.warning(f"CSV 로드 중 예기치 못한 오류: {e}")
        return None

def save_csv_safe(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        st.error(f"CSV 저장 실패: {e}")
        return False

def ensure_q_columns(df: pd.DataFrame, q_count: int = 30):
    """Ensure Q01..Q{q_count} columns exist; if missing, create with NaN (for safety in admin view)."""
    cols = [f"Q{i:02d}" for i in range(1, q_count + 1)]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df, cols

# -----------------------------
# Sidebar: Admin
# -----------------------------
st.sidebar.subheader("🔐 관리자 모드")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

admin_pw = st.sidebar.text_input("관리자 비밀번호", type="password")
if st.sidebar.button("로그인"):
    if admin_pw and os.environ.get("ADMIN_PASSWORD", "") == admin_pw:
        st.session_state.authenticated = True
        st.sidebar.success("인증 성공")
    else:
        st.sidebar.error("인증 실패")

if st.session_state.authenticated:
    df_dl = load_csv_safe(DATA_PATH)
    if df_dl is not None:
        st.sidebar.download_button(
            label="📥 응답 데이터 다운로드 (CSV)",
            data=df_dl.to_csv(index=False).encode("utf-8-sig"),
            file_name="responses_tadt.csv",
            mime="text/csv",
        )
        if st.sidebar.button("🧹 CSV 초기화(백지)"):
            try:
                os.remove(DATA_PATH)
                st.sidebar.success("CSV 삭제 완료 (최초 응답 시 자동 생성됩니다).")
            except FileNotFoundError:
                st.sidebar.info("이미 파일이 없습니다.")
            except Exception as e:
                st.sidebar.error(f"삭제 실패: {e}")
    else:
        st.sidebar.info("저장된 응답이 없습니다. 첫 제출 시 파일이 자동 생성됩니다.")

# -----------------------------
# Q-set (max 30)
# -----------------------------
Q_SET = [
    # A. 감정 전략/어조
    "동일한 정확도라면, 공감적 어조는 사용자 신뢰를 유의하게 높인다.",                               # Q01
    "공감 표현이 근거(팩트·출처) 없이 반복되면 신뢰는 오히려 저하된다.",                              # Q02
    "중립·정보 중심 어조는 단기 효율에는 유리하지만 장기 관계 신뢰에는 한계가 있다.",                   # Q03
    "감정 톤의 일관성은 정확성의 일관성만큼 신뢰 축적에 중요하다.",                                    # Q04
    # B. 반복/시간 동역학
    "작은 기대 위반이 반복되면 신뢰는 비선형적으로 급격히 무너진다.",                                  # Q05
    "오류 이후 사과·설명·수정 계획 제시는 신뢰 회복을 촉진한다.",                                       # Q06
    "개인 맥락(이력·선호)을 기억하는 AI는 반복 상호작용에서 신뢰를 더 빨리 축적한다.",                   # Q07
    "모델 한계·불확실성의 명시는 과신을 줄이고 지속 신뢰를 높인다.",                                     # Q08
    # C. 설명가능성/통제
    "설명 가능한 근거 제시는 공감 표현보다 무결성 신뢰를 더 강하게 만든다.",                           # Q09
    "사용자가 응답 톤(공감/중립)을 선택·조절할 수 있을 때 신뢰가 높아진다.",                             # Q10
    "사용자 피드백이 학습 루프에 반영된다는 신호가 있을 때 장기 신뢰가 강화된다.",                       # Q11
    # D. 인간화/의인화
    "적정 수준의 인간화 단서(이름·일관된 페르소나)는 신뢰 형성에 도움이 된다.",                        # Q12
    "과도한 인간화·감정 과시는 언캐니 효과로 신뢰를 떨어뜨린다.",                                       # Q13
    "‘사람처럼 보이는가’보다 페르소나의 일관성이 신뢰에 더 중요하다.",                                   # Q14
    # E. 위험도/도메인 원칙
    "고위험·고책임 영역에서는 Human-in-the-Loop가 기본 설계 원칙이어야 한다.",                          # Q15
    "저위험·정형 업무에서는 AI 단독 위임이 효율·품질 모두에서 타당하다.",                                # Q16
    "공감 중심 업무에서는 인간–AI 협업이 인간 단독·AI 단독보다 성과가 높다.",                           # Q17
    # F. 운영 모델/프로세스
    "명확한 에스컬레이션 규칙(대화 중단→사람 연결)은 사용자 신뢰를 보호한다.",                          # Q18
    "응답 SLA·품질과 정서 적합성을 함께 측정할 때 조직 신뢰가 유지된다.",                               # Q19
    "데이터 최소수집·프라이버시 보장은 감정 데이터 활용의 필수 신뢰 조건이다.",                         # Q20
    "편향·공정성 완화 노력의 가시화는 무결성 신뢰를 강화한다.",                                         # Q21
    # G. 인사/역량/교육
    "조직은 ‘작성자’보다 검수자/큐레이터/상황조절자 역량을 중시하도록 직무를 재설계해야 한다.",           # Q22
    "공감 커뮤니케이션과 AI 리터러시의 동시 훈련이 협업 성과를 극대화한다.",                             # Q23
    "보상·평가가 정확성뿐 아니라 정서 적합성을 반영할 때 채택이 촉진된다.",                              # Q24
    # H. 서비스 혁신/전략 매트릭스
    "예측 중심 직무군은 ‘위임 전략’이 기본 원칙이어야 한다.",                                           # Q25
    "공감 중심 직무군은 ‘협업 전략’이 기본 원칙이어야 한다.",                                           # Q26
    "고객센터에서는 AI 초안+인간 최종 검수(하이브리드)가 비용과 만족도를 동시에 개선한다.",               # Q27
    "의료에서는 공감형 초진·설명 + 의사 판단 결합이 안전성과 신뢰를 보장한다.",                           # Q28
    "교육에서는 AI의 개별 피드백 + 교사의 정서 코칭이 학습 지속성을 높인다.",                            # Q29
    "사용자 세분화(도구지향/관계지향)에 따라 톤과 자동화 수준을 차등 제공해야 한다.",                    # Q30
]

# Axis weights for strategy mapping
AXIS_WEIGHTS = {
    "Empathy": {
        "Q01": +1, "Q02": -1, "Q04": +1,
        "Q12": +1, "Q13": -1, "Q14": +1,
        "Q17": +1, "Q23": +1, "Q24": +1, "Q29": +1,
        "Q28": +1
    },
    "Predictive": {
        "Q03": +1, "Q09": +1, "Q10": +1, "Q11": +1,
        "Q16": +1, "Q25": +1, "Q27": +1,
        "Q15": -1
    },
    "Delegation": {
        "Q16": +1, "Q25": +1, "Q27": +1,
        "Q15": -1, "Q17": -1, "Q26": -1, "Q28": -1, "Q29": -1
    },
    "Collaboration": {
        "Q15": +1, "Q17": +1, "Q18": +1, "Q19": +1,
        "Q26": +1, "Q28": +1, "Q29": +1,
        "Q16": -1, "Q25": -1
    },
}

# Required domain scenarios
SCENARIOS = {
    "고객센터": "당신은 고객 불만을 처리하는 상담사입니다. 반복적으로 비슷한 불만을 접수하며, AI 도우미가 초안을 제시합니다. "
               "AI의 감정 톤(공감/중립)과 설명(근거·사실)이 신뢰와 효율에 어떤 영향을 줄지 상상해 주세요.",
    "의료": "당신은 환자와 보호자에게 검사 결과를 설명하는 역할을 합니다. AI가 먼저 설명 초안을 제공하고, "
           "당신이 보완·결정합니다. 공감적 설명과 불확실성 고지가 신뢰에 미치는 영향을 고려해 주세요.",
    "교육": "당신은 학습자에게 피드백을 제공하는 교사입니다. AI가 개별 피드백을 제안하고, "
           "당신이 정서 코칭을 결합합니다. 반복 상호작용에서 신뢰가 어떻게 변할지 상상해 주세요.",
}

LIKERT = ["전혀 동의하지 않음(1)", "동의하지 않음(2)", "보통(3)", "동의함(4)", "매우 동의함(5)"]
LIKERT_MAP = {
    "전혀 동의하지 않음(1)": 1,
    "동의하지 않음(2)": 2,
    "보통(3)": 3,
    "동의함(4)": 4,
    "매우 동의함(5)": 5
}

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["✍️ 설문 응답(필수 정보 포함)", "📊 유형/전략 매핑", "🧠 결과 요약 및 권고"])

# -----------------------------
# Survey Tab (Required fields)
# -----------------------------
with tab1:
    st.subheader("✍️ Q-Method Likert 설문")
    st.markdown("- 시나리오 **필수 선택** · 이메일 **필수 입력**")

    domain_options = ["— 시나리오 선택 —"] + list(SCENARIOS.keys())
    domain = st.selectbox("시나리오를 선택해 주세요 (필수)", options=domain_options, index=0)

    if domain in SCENARIOS:
        st.info(SCENARIOS[domain])
    else:
        st.warning("시나리오를 선택하면 설명이 표시됩니다.")

    email = st.text_input("이메일을 입력해 주세요 (필수)")

    with st.form("likert_form"):
        answers = {}
        for i, stmt in enumerate(Q_SET, start=1):
            qid = f"Q{i:02d}"
            sel = st.radio(f"{i}. {stmt}", LIKERT, horizontal=True, key=f"r_{qid}")
            answers[qid] = LIKERT_MAP[sel]
        submitted = st.form_submit_button("제출")

    if submitted:
        # 필수 조건 체크
        valid_domain = domain in SCENARIOS
        valid_email = is_valid_email(email)

        if not valid_domain and not valid_email:
            st.error("시나리오를 선택하고, 올바른 이메일을 입력해 주세요.")
        elif not valid_domain:
            st.error("시나리오를 선택해 주세요.")
        elif not valid_email:
            st.error("올바른 이메일 형식을 입력해 주세요 (예: name@example.com).")
        else:
            try:
                row = {
                    **answers,
                    "domain": domain,
                    "email": email.strip(),
                    "ts": datetime.datetime.now().isoformat()
                }
                df_old = load_csv_safe(DATA_PATH)
                if df_old is None:
                    df_all = pd.DataFrame([row])
                else:
                    df_old, _ = ensure_q_columns(df_old, q_count=len(Q_SET))
                    df_all = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)

                if save_csv_safe(df_all, DATA_PATH):
                    st.success("응답이 저장되었습니다. 감사합니다! (최초 실행이라면 파일이 새로 생성되었습니다)")
            except Exception as e:
                st.error(f"응답 저장 중 오류가 발생했습니다: {e}")

# -----------------------------
# Analysis Helpers
# -----------------------------
def _eigen_k(df_numeric):
    """Compute number of factors by Kaiser (eig >= 1)."""
    try:
        corr = np.corrcoef(df_numeric.T)
        if not np.isfinite(corr).all():
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        eigvals = np.linalg.eigvalsh(corr)
        return int(np.sum(eigvals >= 1.0)), eigvals[::-1]
    except Exception:
        return 2, np.array([1.1, 1.05])

def factor_or_pca(df_numeric, n_factors=None):
    """Fit FactorAnalyzer (if available) else PCA; return loadings DataFrame and model name."""
    cols = df_numeric.columns
    if n_factors is None or n_factors < 1:
        k, _ = _eigen_k(df_numeric)
        n_factors = max(2, min(5, k))  # clamp 2..5

    try:
        if _FA_AVAILABLE:
            fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
            fa.fit(df_numeric)
            load = pd.DataFrame(fa.loadings_, index=cols, columns=[f"Type{i+1}" for i in range(n_factors)])
            return load, "FA", n_factors
        else:
            pca = PCA(n_components=n_factors, random_state=42)
            pca.fit(df_numeric)
            load = pd.DataFrame(pca.components_.T, index=cols, columns=[f"Type{i+1}" for i in range(n_factors)])
            return load, "PCA", n_factors
    except Exception as e:
        st.warning(f"요인/주성분 분석 실패: {e}. 간이 2성분 PCA로 대체합니다.")
        try:
            pca = PCA(n_components=2, random_state=42)
            pca.fit(df_numeric)
            load = pd.DataFrame(pca.components_.T, index=cols, columns=["Type1", "Type2"])
            return load, "PCA-fallback", 2
        except Exception as e2:
            st.error(f"PCA 대체도 실패: {e2}")
            load = pd.DataFrame({"Type1": np.ones(len(cols))}, index=cols)
            return load, "Mock", 1

def axis_scores_from_loadings(loadings: pd.DataFrame, axis_weights: dict) -> pd.DataFrame:
    """Compute axis scores per Type (Empathy, Predictive, Delegation, Collaboration)."""
    out = []
    for type_col in loadings.columns:
        row = {"Type": type_col}
        for axis, weights in axis_weights.items():
            num = 0.0
            den = 0.0
            for qid, w in weights.items():
                if qid in loadings.index:
                    val = loadings.loc[qid, type_col]
                    if pd.notnull(val):
                        num += val * w
                        den += abs(w)
            row[axis] = (num / den) if den > 0 else 0.0
        out.append(row)
    return pd.DataFrame(out).set_index("Type")

def plot_strategy_scatter(df_axes: pd.DataFrame, x, y, title):
    try:
        fig, ax = plt.subplots()
        ax.axhline(0, ls="--", lw=1)
        ax.axvline(0, ls="--", lw=1)
        ax.scatter(df_axes[x], df_axes[y])
        for label, xy in df_axes[[x, y]].iterrows():
            ax.annotate(label, (xy[x], xy[y]), xytext=(5,5), textcoords="offset points")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"그래프 렌더링 실패: {e}")

def recommendations_for_type(emp, pred, delg, coll):
    recs = []
    if pred >= 0.2 and coll <= 0 and delg >= 0.2:
        recs.append("전략: **위임 우선 (Predictive/Delegation High)** — 저위험·정형 업무 자동화, 인간 검수 최소화")
        recs.append("운영모델: 자동 라우팅·자동응답, 예외시 에스컬레이션")
        recs.append("인사전략: 데이터·프로세스 설계 역량 강화, 모니터링/품질관리 직무 육성")
        recs.append("서비스 혁신: 셀프서비스·FAQ 자동화, 초안 자동 생성 파이프라인")
    if coll >= 0.2 and emp >= 0.2:
        recs.append("전략: **협업 우선 (Empathy/Collaboration High)** — HIL·Co-pilot 중심 운영")
        recs.append("운영모델: 공감형 초안 + 인간 최종, 정서 적합성 KPI 도입, 명확한 에스컬레이션")
        recs.append("인사전략: 공감 커뮤니케이션 + AI 리터러시 동시 훈련, 큐레이션·설명 역량 강화")
        recs.append("서비스 혁신: 고객센터/의료/교육 하이브리드 블루프린트(초안→검수→설명)")
    if (pred > emp) and (coll > delg):
        recs.append("혼합 전략: **성능은 예측 지향, 오케스트레이션은 협업형** — 모델 설명/근거 제공과 인간 조정 결합")
    if (emp > pred) and (delg > coll):
        recs.append("주의: **감정 의존 + 위임** 조합은 위험 — 공감 오작동 시 신뢰 붕괴. HIL 강화 필요")
    if not recs:
        recs.append("중립 전략: 도메인·리스크에 따라 위임/협업 혼합. HIL 스위치와 품질/정서 KPI 동시 운영")
    return recs

# -----------------------------
# Analysis Tab
# -----------------------------
with tab2:
    st.subheader("📊 유형 도출(Q) 및 전략 매핑")
    df = load_csv_safe(DATA_PATH)
    if df is None:
        st.info("아직 수집된 응답이 없습니다. 설문 탭에서 먼저 응답을 수집하세요.")
    else:
        df, qcols = ensure_q_columns(df, q_count=len(Q_SET))
        qcols = [c for c in qcols if c in df.columns]
        df_q = df[qcols].copy()

        # 유효 응답자(문항 60% 이상 응답)만 분석
        valid_mask = df_q.notna().sum(axis=1) >= max(5, int(len(qcols) * 0.6))
        df_q = df_q[valid_mask]

        n = len(df_q)
        st.write(f"현재 유효 응답자 수: **{n}명**")
        if n < MIN_N_FOR_ANALYSIS:
            st.warning(f"분석에는 최소 {MIN_N_FOR_ANALYSIS}명의 유효 응답이 필요합니다.")
        elif df_q.shape[1] < 3:
            st.warning("유효 문항 수가 부족합니다.")
        else:
            try:
                df_num = df_q.astype(float) + np.random.normal(0, 1e-3, df_q.shape)  # 작은 노이즈
                df_num = (df_num - df_num.mean()) / (df_num.std(ddof=0) + 1e-8)       # 표준화

                loadings, method_name, k = factor_or_pca(df_num)
                st.write(f"분석 방식: **{method_name}**, 추출된 유형 수: **{k}**")
                st.dataframe(loadings.style.background_gradient(cmap='Blues', axis=None))

                axes_df = axis_scores_from_loadings(loadings, AXIS_WEIGHTS)
                st.markdown("### 📐 유형별 축 점수 (정규화 전)")
                st.dataframe(axes_df)

                # -1..1 정규화
                axes_norm = axes_df.copy()
                for c in axes_norm.columns:
                    vmax = max(1e-6, axes_norm[c].abs().max())
                    axes_norm[c] = axes_norm[c] / vmax
                st.markdown("### 📐 유형별 축 점수 (정규화, -1..1)")
                st.dataframe(axes_norm)

                st.markdown("#### 전략 매트릭스 1: 예측 vs 공감")
                plot_strategy_scatter(axes_norm, "Predictive", "Empathy", "예측(Predictive) vs 공감(Empathy)")

                st.markdown("#### 전략 매트릭스 2: 위임 vs 협업")
                plot_strategy_scatter(axes_norm, "Delegation", "Collaboration", "위임(Delegation) vs 협업(Collaboration)")

                st.download_button(
                    "📥 유형-축 점수 다운로드 (CSV)",
                    data=axes_norm.to_csv().encode("utf-8-sig"),
                    file_name="type_axis_scores.csv",
                    mime="text/csv"
                )

                st.markdown("---")
                st.markdown("### 🧭 유형별 권고안 (운영모델 / 인사전략 / 서비스 혁신)")
                rec_rows = []
                for t, r in axes_norm.iterrows():
                    recs = recommendations_for_type(
                        emp=r["Empathy"], pred=r["Predictive"],
                        delg=r["Delegation"], coll=r["Collaboration"]
                    )
                    st.markdown(f"**{t}**")
                    for bullet in recs:
                        st.markdown(f"- {bullet}")
                    rec_rows.append({
                        "Type": t,
                        "Empathy": r["Empathy"],
                        "Predictive": r["Predictive"],
                        "Delegation": r["Delegation"],
                        "Collaboration": r["Collaboration"],
                        "Recommendations": " | ".join(recs)
                    })
                rec_df = pd.DataFrame(rec_rows)
                st.download_button(
                    "📥 유형별 권고안 다운로드 (CSV)",
                    data=rec_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="type_recommendations.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")

# -----------------------------
# Summary Tab
# -----------------------------
with tab3:
    st.subheader("🧠 결과 요약 및 활용 가이드")
    st.markdown("""
    - 본 도구는 Q-method를 **Likert**로 수집하고, 요인/주성분 분석으로 **인식 유형(Type)**을 도출합니다.
    - 유형별 축 점수(예측/공감/위임/협업)로 두 개의 **전략 매트릭스**에 매핑합니다.
    - 각 유형에 대해 **운영모델(에스컬레이션·정서KPI·HIL), 인사전략(역량·교육·보상), 서비스 혁신(하이브리드 설계)** 권고안을 생성합니다.
    - 시나리오는 **필수** 선택, 이메일은 **필수** 입력입니다.
    """)
