"""
Q-Method (TADT Research) — Q Analyzer
@Author: Prof. Dr. Songhee Kang
@Date: 2025.08.14. 
- 사람 요인화(Q): 응답자 간 상관행렬 → 고유분해 → Varimax 회전 → 유형/배정/상하위 진술
- CSV가 '존재하지만 비어있는' 경우 첫 제출로 같은 파일에 채움
- 제출 시 GitHub에 즉시 커밋(REST API, PyGithub 불요)
- 사이드바: 자동 동기화 토글 / 지금 동기화 버튼 / 관리자 다운로드
"""

import os, re, base64, json, datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Page & Globals
# -----------------------------
st.set_page_config(page_title="Q-Method (TADT Research) Analyzer", layout="wide")
st.title("Q-Method (TADT Research) Analyzer")

DATA_PATH = "responses_tadt.csv"   # 로컬 CSV 경로
MIN_N_FOR_ANALYSIS = 5
TOPK_STATEMENTS = 5
EPS = 1e-8

# -----------------------------
# Optional font
# -----------------------------
try:
    import matplotlib.font_manager as fm
    font_path = "fonts/NanumGothic.ttf"
    if os.path.exists(font_path):
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
except Exception:
    pass

# -----------------------------
# Secrets (GitHub)
# -----------------------------
def _get_secret(path, default=""):
    try:
        cur = st.secrets
        for key in path.split("."):
            cur = cur[key]
        return cur
    except Exception:
        return default

GH_TOKEN   = _get_secret("github.token")
GH_REPO    = _get_secret("github.repo")
GH_BRANCH  = _get_secret("github.branch", "main")
GH_REMOTEP = _get_secret("github.data_path", "responses_tadt.csv")  # 원격 저장 경로
GH_README  = _get_secret("github.readme_path", "README.md")         # (옵션)

# -----------------------------
# Admin (optional)
# -----------------------------
st.sidebar.subheader("🔐 관리자 / 동기화")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

admin_pw = st.sidebar.text_input("관리자 비밀번호 (선택)", type="password")
if st.sidebar.button("로그인"):
    if admin_pw and _get_secret("admin.password") == admin_pw:
        st.session_state.authenticated = True
        st.sidebar.success("인증 성공")
    else:
        st.sidebar.error("인증 실패")

auto_sync = st.sidebar.checkbox("응답 저장 시 GitHub 자동 푸시", value=True)

def _gh_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "streamlit-qmethod-tadt"
    }

def gh_get_sha(owner_repo, path, token, branch):
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(token), params={"ref": branch}, timeout=20)
    if r.status_code == 200:
        try:
            return r.json().get("sha")
        except Exception:
            return None
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"GitHub GET 실패: {r.status_code} {r.text}")

def gh_put_file(owner_repo, path, token, branch, content_bytes, message):
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    b64 = base64.b64encode(content_bytes).decode("ascii")
    sha = gh_get_sha(owner_repo, path, token, branch)
    payload = {"message": message, "content": b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(token), data=json.dumps(payload), timeout=30)
    if r.status_code in (200, 201):
        return True, r.json()
    return False, f"{r.status_code}: {r.text}"

def push_csv_to_github(local_path, remote_path=None, note="Update responses_tadt.csv"):
    if not (GH_TOKEN and GH_REPO):
        return False, "GitHub secrets 누락(github.token, github.repo)"
    if remote_path is None:
        remote_path = GH_REMOTEP
    try:
        with open(local_path, "rb") as f:
            content = f.read()
    except Exception as e:
        return False, f"로컬 CSV 읽기 실패: {e}"
    ok, resp = gh_put_file(GH_REPO, remote_path, GH_TOKEN, GH_BRANCH, content, note)
    return ok, resp

# -----------------------------
# Utils
# -----------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def is_valid_email(s: str) -> bool:
    if not s: return False
    s = s.strip()
    if len(s) > 150: return False
    return bool(EMAIL_RE.match(s))

def load_csv_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def save_csv_safe(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        st.error(f"CSV 저장 실패: {e}")
        return False

def ensure_q_columns(df: pd.DataFrame, q_count: int):
    cols = [f"Q{i:02d}" for i in range(1, q_count + 1)]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df, cols

def zscore_rows(a: np.ndarray):
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, ddof=0, keepdims=True)
    s = np.where(s < EPS, 1.0, s)
    return (a - m) / s

def rank_rows(a: np.ndarray):
    df = pd.DataFrame(a)
    return df.rank(axis=1, method="average", na_option="keep").values

def varimax(Phi, gamma=1.0, q=100, tol=1e-6, seed=42):
    Phi = Phi.copy(); p, k = Phi.shape
    R = np.eye(k); d_old = 0
    for _ in range(q):
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * (Lambda @ np.diag(np.sum(Lambda**2, axis=0))))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol: break
        d_old = d
    return Phi @ R, R

def choose_n_factors(eigvals, nmax):
    k = int(np.sum(eigvals >= 1.0))
    return max(2, min(nmax, k))

# -----------------------------
# Q-set
# -----------------------------
Q_SET = [
"공감하는 말투는 AI에 대한 신뢰를 높인다. ",
"반복적 공감 표현은 줄이면 신뢰가 높아진다. ",
"중립적 정보 전달에 공감을 더하면 신뢰가 높아진다. ",
"답변 말투가 일정하면 신뢰가 유지된다. ",
"태도와 기준의 일관성은 신뢰를 높인다. ",
"작은 실수라도 수정이 이루어지면 신뢰가 유지된다. ",
"실수 뒤 사과가 있으면 신뢰 회복에 도움이 된다. ",
"수정 계획 제시는 신뢰 회복에 도움이 된다. ",
"AI가 한계점을 밝히면 신뢰가 유지된다. ",
"AI가 불확실성을 구체적으로 제시하면 신뢰가 유지된다. ",
"피드백이 반영되면 신뢰가 높아진다. ",
"출처 제시는 신뢰를 높인다. ",
"근거 제시는 신뢰를 높인다. ",
"개인정보 최소 사용은 신뢰를 높인다. ",
"감정 관련 데이터 최소 사용은 신뢰를 높인다. ",
"편향을 줄이는 노력은 신뢰를 높인다. ",
"편향 관련 결과 공개는 신뢰를 높인다. ",
"사용자가 말투를 선택할 수 있으면 신뢰가 높아진다. ",
"중요한 결론은 이유 설명이 있으면 수용이 쉽다. ",
"가벼운 인간적 단서는 신뢰를 높인다. ",
"과한 감정 표현은 신뢰를 낮춘다. ",
"적절한 감정 표현은 신뢰에 도움이 된다. ",
"말투의 적절성은 성과 지표가 될 수 있다. ",
"개인 선호나 과거 대화를 기억하는 AI는 신뢰를 높인다. ",
"반복적이고 단순한 일은 AI에게 맡겨도 된다. ",
"예측 중심 업무는 AI 활용이 적합하다. ",
"공감 중심 업무는 인간–AI 협업이 적합하다. ",
"위험이 큰 일은 사람이 최종 확인할 때 신뢰가 유지된다. ",
"언제든 사람에게 연결되는 절차가 있으면 신뢰가 유지된다. ",
"정밀한 검토 역할은 작성보다 더 중요하다. ",
"조율 역할은 작성보다 더 중요하다. ",
"고객 응대에서 AI 초안 활용은 효율을 높인다. ",
"고객 응대에서 사람의 보완은 효율을 높인다. ",
"고객 응대에서 AI 초안 활용은 만족도를 높인다. ",
"고객 응대에서 사람의 보완은 만족도를 높인다. ",
"의료 안내에서 공감 있는 설명은 신뢰를 유지한다. ",
"의료 안내에서 사람의 최종 판단은 신뢰를 유지한다. ",
"교육 피드백에서 AI의 개별화 제안은 효과를 높인다. ",
"교육 피드백에서 교사의 정서 코칭은 효과를 높인다. "
]
Q_COLS = [f"Q{i:02d}" for i in range(1, len(Q_SET)+1)]
LIKERT = ["전혀 동의하지 않음(1)", "동의하지 않음(2)", "보통(3)", "동의함(4)", "매우 동의함(5)"]
LIKERT_MAP = {"전혀 동의하지 않음(1)":1,"동의하지 않음(2)":2,"보통(3)":3,"동의함(4)":4,"매우 동의함(5)":5}

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["✍️ 설문 수집(이메일 필수)", "📊 사람 요인화(Q) 분석", "☁️ GitHub 동기화 로그"])

# -----------------------------
# Tab1: Survey
# -----------------------------
with tab1:
    # ── 조사 안내 블록 (Streamlit) ──────────────────────────────────────────────────
    st.markdown("""
    ### 📢 **조사 안내**
    
    **주관:** 한국공학대학교 · 경희대학교 · 국민대학교  
    **소요 시간:** 약 **5–7분** │ **응답 형식:** 5점 리커트(30문항)
    
    ---
    
    #### 🎯 **연구 취지**
    생성형 AI 시대에 공감(감정적 적합성)과 예측(정보 정확성)의 균형, 그리고 **위임**과 **협업**의 경계가 어떻게 설정되어야 하는지를 탐색합니다.  
    본 연구는 TADT(Tech-Affective Dynamics Theory)에 기반하여, 진술문 응답을 통해 참가자별 **인식 유형**을 도출하고, 그 유형을 **예측↔공감 / 위임↔협업** 전략 매트릭스에 매핑합니다.  
    이를 통해 **신뢰가 축적·붕괴되는 조건**과 **지속 가능한 인간–AI 협업 원칙**을 제안합니다.
    
    #### 📝 **참여 안내**
    - 30개 짧은 진술문에 대해 *전혀 동의하지 않음(1) ~ 매우 동의함(5)* 중 하나를 선택해 주세요. 중립적이거나 잘 몰라서 판단을 유보하고 싶은 경우에는 중간(3)으로 주로 유지해주세요. 결과적으로는 중간값이 가장 많도록 선택해주세요. 
    - 응답은 **익명 분석**을 원칙으로 하며, **이메일은 중복 제거 및 사후 안내**(예: 결과 공지, 보상 고지)에만 사용합니다.
    
    """)
    
    with st.expander("🔒 법적·윤리 안내 (펼쳐 보기)", expanded=False):
        st.markdown("""
    - **통계 목적 사용 원칙**: 응답은 **통계작성 및 학술연구 목적**에 한하여 사용되며, 법령에서 정한 경우를 제외하고 **제3자에게 제공되지 않습니다**.  
    - **개인정보 최소수집·분리보관**: 수집 항목은 설문 응답과 이메일 주소(중복 식별·사후 안내용)입니다. 이메일은 **분리 보관**되며 분석 자료에는 포함되지 않습니다.  
    - **익명 처리**: 분석 단계에서는 개인을 식별할 수 없도록 **비식별화/익명 처리**합니다.  
    - **자발적 참여·철회**: 참여는 자발적이며, **언제든 중단 또는 철회**할 수 있습니다(불이익 없음).  
    - **보관·파기**: 연구윤리 지침 및 관련 법령을 준수하여, 연구 종료 후 **정해진 보관기간**이 경과하면 안전하게 **파기**합니다.  
    - **관련 법령 준수**: 본 조사는 **통계 관련 법령** 및 **개인정보 보호 관련 법령**을 준수합니다.
    """)
    
    st.markdown("""
    ---
    **🧑‍🤝‍🧑 공동 연구기관:** **한국공학대학교 · 경희대학교 · 국민대학교**
    """)
    # ────────────────────────────────────────────────────────────────────────────────

    st.subheader("응답 입력 (이메일 필수)")
    email = st.text_input("이메일(필수) — 후속 패널 조사/보상 안내용으로, 관련법에 의거 사용목적에 따라 활용후 폐기됩니다.")
    with st.form("likert_form"):
        answers = {}
        for i, stmt in enumerate(Q_SET, start=1):
            qid = f"Q{i:02d}"
            sel = st.radio(f"{i}. {stmt}", LIKERT, horizontal=True, key=f"r_{qid}")
            answers[qid] = LIKERT_MAP[sel]
        submitted = st.form_submit_button("제출")

    if submitted:
        if not is_valid_email(email):
            st.error("올바른 이메일을 입력해 주세요. (예: name@example.com)")
        else:
            try:
                row = {**answers, "email": email.strip(), "ts": datetime.datetime.now().isoformat()}
                file_exists = os.path.exists(DATA_PATH)
                file_empty  = file_exists and os.path.getsize(DATA_PATH) == 0

                df_old = load_csv_safe(DATA_PATH)
                if df_old is None:
                    df_all = pd.DataFrame([row])
                else:
                    df_old, _ = ensure_q_columns(df_old, q_count=len(Q_SET))
                    df_all = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)

                if save_csv_safe(df_all, DATA_PATH):
                    msg = "응답을 저장했습니다. "
                    if file_exists and file_empty: msg += "(빈 파일에 첫 응답 기록)"
                    st.success(msg)

                    # 🔁 GitHub 자동 동기화
                    if auto_sync:
                        ok, resp = push_csv_to_github(DATA_PATH, GH_REMOTEP,
                                                      note=f"Update {GH_REMOTEP} at {datetime.datetime.now().isoformat()}")
                        if ok:
                            st.success("GitHub에 동기화되었습니다.")
                        else:
                            st.warning(f"GitHub 동기화 실패: {resp}")
                    else:
                        st.info("자동 동기화가 꺼져 있습니다. 사이드바에서 '지금 동기화'를 누르세요.")
            except Exception as e:
                st.error(f"저장 중 오류: {e}")

# -----------------------------
# Person-Q Analysis Helpers
# -----------------------------
def person_q_analysis(df_q: pd.DataFrame,
                      corr_metric: str = "Pearson",
                      n_factors: int | None = None,
                      rotate: bool = True):
    M = df_q.values.astype(float)
    if corr_metric.lower().startswith("spear"):
        M_proc = rank_rows(M)
        M_proc = zscore_rows(M_proc)
    else:
        M_proc = zscore_rows(M)

    R = np.corrcoef(M_proc, rowvar=True)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]

    nmax = max(2, min(6, R.shape[0]-1))
    if n_factors is None or n_factors < 1:
        n_factors = choose_n_factors(eigvals, nmax)
    else:
        n_factors = max(2, min(nmax, int(n_factors)))

    L = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0))
    L_rot = varimax(L)[0] if rotate else L

    arrays = []
    for k in range(n_factors):
        w = np.clip(L_rot[:, k], a_min=0.0, a_max=None)
        if w.sum() <= EPS: w = np.abs(L_rot[:, k])
        w = w / (w.sum() + EPS)
        arr_k = w @ M_proc
        arrays.append(arr_k)
    arrays = np.vstack(arrays)
    return L_rot, eigvals, R, arrays

def assign_types(loadings: np.ndarray, emails: list[str], thr: float = 0.40, sep: float = 0.10):
    N, K = loadings.shape
    max_idx = loadings.argmax(axis=1)
    max_val = loadings.max(axis=1)
    sorted_vals = np.sort(loadings, axis=1)[:, ::-1]
    second = sorted_vals[:,1] if K >= 2 else np.zeros(N)
    assigned = (max_val >= thr) & ((max_val - second) >= sep)
    rows = []
    for i in range(N):
        rows.append({
            "email": emails[i] if i < len(emails) else f"id_{i}",
            "Type": f"Type{max_idx[i]+1}",
            "MaxLoading": float(max_val[i]),
            "Second": float(second[i]),
            "Assigned": bool(assigned[i])
        })
    return pd.DataFrame(rows)

def top_bottom_statements(factor_arrays: np.ndarray, topk=5):
    K, P = factor_arrays.shape
    tb = []
    for k in range(K):
        z = factor_arrays[k]
        top_idx = np.argsort(z)[::-1][:topk]
        bot_idx = np.argsort(z)[:topk]
        tb.append((top_idx, bot_idx, z))
    return tb

# -----------------------------
# Tab2: Person-Q Analysis
# -----------------------------
with tab2:
    st.subheader("사람 요인화(Q) 분석")
    df = load_csv_safe(DATA_PATH)
    if df is None:
        st.info("아직 수집된 응답이 없습니다. (빈 파일이면 먼저 설문 제출)")
    else:
        df, _ = ensure_q_columns(df, q_count=len(Q_SET))
        df_q = df[Q_COLS].copy()
        mask = df_q.notna().sum(axis=1) >= int(0.6*len(Q_COLS))
        df_q = df_q[mask]
        emails = df.loc[mask, "email"].fillna("").astype(str).tolist()

        st.write(f"유효 응답자 수: **{len(df_q)}명**")
        if len(df_q) < MIN_N_FOR_ANALYSIS:
            st.warning(f"분석에는 최소 {MIN_N_FOR_ANALYSIS}명의 응답이 필요합니다.")
        else:
            with st.expander("⚙️ 분석 옵션", expanded=True):
                colA, colB, colC = st.columns(3)
                with colA:
                    corr_metric = st.selectbox("상관계수", ["Pearson", "Spearman"], index=0)
                with colB:
                    n_f_override = st.number_input("요인 수(선택, 0=자동)", min_value=0, max_value=6, value=0, step=1)
                    n_factors = None if n_f_override == 0 else int(n_f_override)
                with colC:
                    rotate = st.checkbox("Varimax 회전", value=True)

                thr = st.slider("유형 배정 임계값(최대 적재치)", 0.20, 0.70, 0.40, 0.05)
                sep = st.slider("1등-2등 적재치 최소 격차", 0.00, 0.50, 0.10, 0.05)

            try:
                loadings, eigvals, R, arrays = person_q_analysis(df_q, corr_metric, n_factors, rotate)
                K = loadings.shape[1]

                st.markdown(f"**추출 요인 수: {K}**")
                load_df = pd.DataFrame(loadings, columns=[f"Type{i+1}" for i in range(K)])
                load_df.insert(0, "email", emails)
                st.dataframe(load_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

                assign_df = assign_types(loadings, emails, thr=thr, sep=sep)
                st.markdown("### 참가자 유형 배정")
                st.dataframe(assign_df, use_container_width=True)
                st.write("유형별 인원수:", assign_df[assign_df["Assigned"]].groupby("Type").size().to_dict())

                st.download_button(
                    "📥 참가자-유형 배정 CSV",
                    data=assign_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="person_type_assignments.csv",
                    mime="text/csv"
                )

                arrays_df = pd.DataFrame(arrays, columns=Q_COLS, index=[f"Type{i+1}" for i in range(K)])
                st.markdown("### 유형별 factor array (진술 z-프로파일)")
                st.dataframe(arrays_df, use_container_width=True)
                st.download_button(
                    "📥 유형별 factor array CSV",
                    data=arrays_df.to_csv().encode("utf-8-sig"),
                    file_name="type_factor_arrays.csv",
                    mime="text/csv"
                )

                st.markdown(f"### 유형별 상/하위 진술 Top {TOPK_STATEMENTS}")
                tb = top_bottom_statements(arrays, topk=TOPK_STATEMENTS)
                for i, (top_idx, bot_idx, z) in enumerate(tb, start=1):
                    with st.expander(f"Type{i} 상/하위 진술", expanded=True if i==1 else False):
                        st.markdown("**상위(+) 진술**")
                        for j in top_idx:
                            st.write(f"- Q{j+1:02d} (z={z[j]:.2f}) : {Q_SET[j]}")
                        st.markdown("**하위(−) 진술**")
                        for j in bot_idx:
                            st.write(f"- Q{j+1:02d} (z={z[j]:.2f}) : {Q_SET[j]}")

            except Exception as e:
                st.error(f"사람 요인화 분석 중 오류: {e}")

# -----------------------------
# Tab3: GitHub Sync Log / Manual Push
# -----------------------------
with tab3:
    st.subheader("GitHub 동기화")
    if not (GH_TOKEN and GH_REPO):
        st.warning("Secrets에 github.token, github.repo 설정이 필요합니다.")
        st.code("""
[github]
token = "ghp_..."
repo  = "owner/repo"
branch = "main"
data_path = "data/responses_tadt.csv"
        """, language="toml")
    else:
        st.success(f"원격: {GH_REPO} @ {GH_BRANCH}\n경로: {GH_REMOTEP}")

    if st.button("지금 동기화(수동)"):
        if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
            ok, resp = push_csv_to_github(DATA_PATH, GH_REMOTEP,
                                          note=f"Manual sync {GH_REMOTEP} at {datetime.datetime.now().isoformat()}")
            if ok:
                st.success("GitHub에 동기화되었습니다.")
                st.json(resp)
            else:
                st.error(f"동기화 실패: {resp}")
        else:
            st.error("로컬 CSV가 없거나 비어있습니다. 먼저 설문을 제출해 CSV를 생성하세요.")
