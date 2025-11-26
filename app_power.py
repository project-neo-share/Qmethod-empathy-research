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

DATA_PATH = "responses_power1.csv"   # 로컬 CSV 경로
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
GH_REMOTEP = _get_secret("github.data_path", "responses_power1.csv")  # 원격 저장 경로
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
"C1. 우리나라 에너지 정책은 공정한 절차에 따라 추진되고 있다."
"C2. 정부의 에너지 정책은 다양한 이해관계자의 의견을 반영해 사회적 합의가 충분히 이루어지고 있다."
"C3. 에너지 정책은 전문적인 영역이므로 전문가가 결정하도록 해야 한다."
"C4. 우리나라 에너지 정책은 정치적 성향과 이념의 영향을 받고 있다."
"C5. 정부가 원자력 정책을 공정하고 책임감 있게 추진하고 있다."
"C6. 정부가 원자력 발전 정책에 대한 정보를 투명하게 공개하고 있다."
"C7. 원자력 발전 정책에 대한 정보를 신뢰할 수 있다."
"C8. 원자력 정책은 다양한 이해관계자의 의견을 충분히 반영하고 있다."
"C9. 정부가 신재생에너지 정책을 공정하고 책임감 있게 추진하고 있다."
"C10. 정부가 신재생에너지 정책에 대한 정보를 투명하게 공개하고 있다."
"C11. 신재생에너지 정책은 다양한 이해관계자의 의견을 충분히 반영하고 있다."
"C12. 원자력발전은 우리나라에 꼭 필요한 발전원이다."
"C13. 원자력발전은 에너지 비용을 낮추는데 효과적이라고 생각한다."
"C14. 원자력발전은 나에게 경제적인 이득을 줄 것이다."
"C15. 원자력발전은 에너지 자립도를 강화하는데 중요한 역할을 한다."
"C16. 향후 원자력발전의 비중을 지금보다 더 높여야 한다."
"C17. 우리나라 원자력발전은 안전하다고 생각한다."
"C18. 사용후핵연료 최종처분시설은 안전하다고 생각한다."
"C19. 원자력 발전은 온실가스 배출을 줄이는데 효과적이다."
"C20. 원자력 발전은 환경에 미치는 부정적 영향이 다른 발전원에 비해 낮다."
"C21. 원자력 발전은 지속가능한 에너지원이다."
"C22. 우리나라의 원자력발전기술은 타 발전기술 대비 우수하다."
"C23. 우리나라 원자력발전의 기술경쟁력을 지속적으로 키워야한다."
"C24. 우리나라 원자력발전의 기술우위로 우리 경제가 더 발전할 것이다."
"C25. 후쿠시마 오염수 방류는 나를 불안하게 만든다."
"C26. 후쿠시마 오염수 방류는 내 건강에 유의미한 악영향을 미칠 것이다."
"C27. 나는 후쿠시마 오염수 방류 이후 수산물을 더 적게 소비할 것이다."
"C28. 신재생에너지는 에너지 비용을 낮추는데 효과적이라고 생각한다."
"C29. 신재생에너지는 나에게 경제적인 이득을 줄 것이다."
"C30. 신재생에너지는 에너지 자립도를 강화하는데 중요한 역할을 한다."
"C31. 신재생에너지발전은 온실가스 배출을 줄이는데 효과적이다."
"C32. 신재생에너지발전은 환경에 미치는 부정적 영향이 다른 발전원에 비해 낮다."
"C33. 신재생에너지발전은 지속가능한 에너지원이다."
"C34. 우리나라의 신재생에너지기술은 타 발전기술 대비 우수하다."
"C35. 우리나라 신재생에너지의 기술경쟁력을 지속적으로 키워야한다."
"C36. 원자력 정책 결정과정에서 형식적인 여론수렴은 갈등을 악화할 수 있다."
"C37. 원자력 발전소 건설에서 금전 보상만으로는 안전·신뢰에 따른 수용성 문제를 해결할 수 없다."
"C38. 원전의 잦은 정지와 고장은 사고 위험을 높일 것이다."
"C39. 원자력발전의 연료주기와 발전소 해체까지 고려하면 환경영향이 더 커질 것이다."
"C40. 오염수 방류 위험이 있는 지역에선 어업·관광 생계 전환에 대한 지원이 필요하다."
"C41. 원자력 발전 규제기관의 독립성이 부족하면 어떤 대책도 신뢰하기 어렵다."
"C42. 소형모듈형원자력발전(SMR)은 상업 운영 데이터가 부족해 신뢰하기 어렵다."
"C43. 에너지 안보를 위해 핵연료·부품 공급망 지정학 리스크를 상시 점검하고 대체 경로를 확보해야 한다."
"C44. 원자력 발전 확대를 위해 기존 부지 내부에 대체·증설을 우선 검토해야한다."
"C45. 원자력 발전 건설에 앞서 원자력 발전에 대한 사고나 피해에 대한 책임보험·배상 절차를 사전 확정해야 수용성을 높일 수 있다."
]
Q_COLS = [f"Q{i:02d}" for i in range(1, len(Q_SET)+1)]
# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs([ "📊 사람 요인화(Q) 분석", "☁️ GitHub 동기화 로그"])
   
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
with tab1:
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
with tab2:
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
