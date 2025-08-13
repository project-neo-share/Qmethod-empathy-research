"""
Q-Method (TADT) — Person-Centred (Q) Analyzer + GitHub Auto Push

- 사람 요인화(Q): 응답자 간 상관행렬 → 고유분해 → Varimax 회전 → 유형/배정/상하위 진술
- CSV가 '존재하지만 비어있는' 경우 첫 제출로 같은 파일에 채움
- 제출 시 GitHub에 즉시 커밋(REST API, PyGithub 불요)
- 사이드바: 자동 동기화 토글 / 지금 동기화 버튼 / 관리자 다운로드

Secrets(.streamlit/secrets.toml):
[github]
token = "ghp_..."         # repo 또는 public_repo scope
repo = "owner/repo"
branch = "main"
data_path = "data/responses_tadt.csv"
readme_path = "README.md"  # (옵션)

[admin]
password = "secret123"     # (옵션)
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
st.set_page_config(page_title="Q-Method (TADT) — Person Q + GitHub", layout="wide")
st.title("Q-Method (TADT) — 사람 요인화(Q) + GitHub 동기화")

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
    "동일한 정확도라면, 공감적 어조는 사용자 신뢰를 유의하게 높인다.",
    "공감 표현이 근거(팩트·출처) 없이 반복되면 신뢰는 오히려 저하된다.",
    "중립·정보 중심 어조는 단기 효율에는 유리하지만 장기 관계 신뢰에는 한계가 있다.",
    "감정 톤의 일관성은 정확성의 일관성만큼 신뢰 축적에 중요하다.",
    "작은 기대 위반이 반복되면 신뢰는 비선형적으로 급격히 무너진다.",
    "오류 이후 사과·설명·수정 계획 제시는 신뢰 회복을 촉진한다.",
    "개인 맥락(이력·선호)을 기억하는 AI는 반복 상호작용에서 신뢰를 더 빨리 축적한다.",
    "모델 한계·불확실성의 명시는 과신을 줄이고 지속 신뢰를 높인다.",
    "설명 가능한 근거 제시는 공감 표현보다 무결성 신뢰를 더 강하게 만든다.",
    "사용자가 응답 톤(공감/중립)을 선택·조절할 수 있을 때 신뢰가 높아진다.",
    "사용자 피드백이 학습 루프에 반영된다는 신호가 있을 때 장기 신뢰가 강화된다.",
    "적정 수준의 인간화 단서(이름·일관된 페르소나)는 신뢰 형성에 도움이 된다.",
    "과도한 인간화·감정 과시는 언캐니 효과로 신뢰를 떨어뜨린다.",
    "‘사람처럼 보이는가’보다 페르소나의 일관성이 신뢰에 더 중요하다.",
    "고위험·고책임 영역에서는 Human-in-the-Loop가 기본 설계 원칙이어야 한다.",
    "저위험·정형 업무에서는 AI 단독 위임이 효율·품질 모두에서 타당하다.",
    "공감 중심 업무에서는 인간–AI 협업이 인간 단독·AI 단독보다 성과가 높다.",
    "명확한 에스컬레이션 규칙(대화 중단→사람 연결)은 사용자 신뢰를 보호한다.",
    "응답 SLA·품질과 정서 적합성을 함께 측정할 때 조직 신뢰가 유지된다.",
    "데이터 최소수집·프라이버시 보장은 감정 데이터 활용의 필수 신뢰 조건이다.",
    "편향·공정성 완화 노력의 가시화는 무결성 신뢰를 강화한다.",
    "조직은 ‘작성자’보다 검수자/큐레이터/상황조절자 역량을 중시하도록 직무를 재설계해야 한다.",
    "공감 커뮤니케이션과 AI 리터러시의 동시 훈련이 협업 성과를 극대화한다.",
    "보상·평가가 정확성뿐 아니라 정서 적합성을 반영할 때 채택이 촉진된다.",
    "예측 중심 직무군은 ‘위임 전략’이 기본 원칙이어야 한다.",
    "공감 중심 직무군은 ‘협업 전략’이 기본 원칙이어야 한다.",
    "고객센터에서는 AI 초안+인간 최종 검수(하이브리드)가 비용과 만족도를 동시에 개선한다.",
    "의료에서는 공감형 초진·설명 + 의사 판단 결합이 안전성과 신뢰를 보장한다.",
    "교육에서는 AI의 개별 피드백 + 교사의 정서 코칭이 학습 지속성을 높인다.",
    "사용자 세분화(도구지향/관계지향)에 따라 톤과 자동화 수준을 차등 제공해야 한다.",
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
    st.subheader("응답 입력 (이메일 필수)")
    email = st.text_input("이메일(필수) — 후속 패널 조사/보상 안내용")
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
