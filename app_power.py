"""
Q-Method (TADT Research) — Q Analyzer
@Author: Prof. Dr. Songhee Kang
@Date: 2025.08.14. 
Q-정렬 현장 분석 앱 (PARTA/PARTB/PARTC)
- 세트별(Q-정렬) : 요인 추출/사람-요인 적재/유형 배정/상하위 진술
- 공통문항 교차분석 : 스크리+병렬, Procrustes(세트 간 일치도), 설명분산
- 구별진술 : z-array 근사, z-차 유의성, Humphrey’s rule
- 부트스트랩 : 공통문항 요인 안정도(φ 임계 이상 비율)

필요 라이브러리: pandas, numpy, scipy, scikit-learn, matplotlib, openpyxl, streamlit
"""

import os, io, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.stats import norm as zdist

st.set_page_config(page_title="Q-정렬 현장 분석", layout="wide")

# ========================= 공통 상수/유틸 =========================
EMAIL_COL_CAND = ["email","Email","E-mail","respondent","id"]
MIN_N_FOR_ANALYSIS = 20
TOPK_STATEMENTS = 5
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

def _coerce_numeric(df: pd.DataFrame):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _looks_like_qcol(name: str):
    name_l = str(name).strip().lower()
    if any(k in name_l for k in ["email","respondent","id","time","name","timestamp"]):
        return False
    return True

@st.cache_data(show_spinner=False)
def load_excel_parts(file_bytes: bytes, sheet_names=("PARTA","PARTB","PARTC")):
    """엑셀 바이너리 → 세트 dict('A','B','C') with email+문항, 숫자형 문항만."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    parts = {}
    for sid, sname in zip(["A","B","C"], sheet_names):
        if sname not in xls.sheet_names:
            raise ValueError(f"시트 '{sname}' 를 찾을 수 없습니다. 엑셀 시트: {xls.sheet_names}")
        raw = pd.read_excel(xls, sheet_name=sname)
        # email/ID
        email_col = None
        for c in raw.columns:
            if str(c).strip().lower() in [e.lower() for e in EMAIL_COL_CAND]:
                email_col = c; break
        if email_col is None:
            raw["email"] = ""
            email_col = "email"
        # 문항 후보
        q_cols = [c for c in raw.columns if c!=email_col and _looks_like_qcol(c)]
        num = _coerce_numeric(raw[q_cols])
        valid_cols = [c for c in num.columns if num[c].notna().sum()>=3]
        df_q = num[valid_cols].copy()
        df_q.insert(0, "email", raw[email_col].fillna("").astype(str))
        parts[sid] = df_q.reset_index(drop=True)
    return parts

def ensure_q_columns(df: pd.DataFrame, q_count=None):
    """email + 문항열만 반환 & Q_COLS/Q_SET 제공"""
    cols = list(df.columns)
    if not cols or str(cols[0]).lower()!="email":
        df = df.copy()
        df.insert(0, "email", "")
    Q_COLS = [c for c in df.columns if c!="email"]
    if q_count and len(Q_COLS)>q_count:
        Q_COLS = Q_COLS[:q_count]
    Q_SET = [str(c) for c in Q_COLS]
    return df[["email"]+Q_COLS], (Q_COLS, Q_SET)

def standardize_people_rows(X: np.ndarray):
    return (X - X.mean(axis=1, keepdims=True))/ (X.std(axis=1, ddof=1, keepdims=True)+1e-8)

def person_correlation(df_only: pd.DataFrame, metric="Pearson"):
    """사람×사람 상관: 숫자형 열만 사용"""
    dfn = df_only.select_dtypes(include=[np.number]).copy()
    if dfn.shape[1] < 3:
        raise ValueError("상관분석: 숫자형 문항 열이 3개 미만입니다.")
    X = dfn.to_numpy(dtype=float)
    if metric.lower().startswith("spear"):
        X_rank = np.apply_along_axis(lambda v: pd.Series(v).rank(method="average").to_numpy(), 0, X)
        Xs = (X_rank - X_rank.mean(axis=1, keepdims=True)) / (X_rank.std(axis=1, ddof=1, keepdims=True)+1e-8)
    else:
        Xs = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, ddof=1, keepdims=True)+1e-8)
    return np.corrcoef(Xs)

def varimax(Phi, gamma=1.0, q=60, tol=1e-6):
    from numpy import eye, dot
    p,k = Phi.shape
    R = eye(k); d = 0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = np.linalg.svd(dot(Phi.T, (Lambda**3 - (gamma/p)*dot(Lambda, np.diag(np.diag(dot(Lambda.T,Lambda)))))))
        R = dot(u, vh); d = s.sum()
        if d_old!=0 and d/d_old < 1+tol: break
    return dot(Phi, R)

# ========================= 세트별 Q-분석 함수 =========================
def person_q_analysis(df_q: pd.DataFrame, corr_metric="Pearson", n_factors=None, rotate=True):
    df_only = df_q.drop(columns=["email"], errors="ignore")
    R = person_correlation(df_only, metric=corr_metric)
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:,idx]
    # 자동 요인 수(고유값>1, 2~6로 제한)
    if not n_factors or n_factors<=0:
        n_factors = int(np.sum(eigvals > 1.0))
        n_factors = max(2, min(6, n_factors))
    loadings = eigvecs[:, :n_factors]*np.sqrt(eigvals[:n_factors])  # 사람×요인
    # 문항 z-array 근사
    X = df_only.to_numpy(dtype=float)
    Z_items = (X - X.mean(axis=0))/ (X.std(axis=0, ddof=1)+1e-8)  # 사람×문항
    arrays = []
    for j in range(n_factors):
        w = loadings[:,j]
        idx_top = np.argsort(np.abs(w))[::-1][:max(5, int(0.1*len(w)))]
        z_j = (Z_items[idx_top].T @ w[idx_top]) / (np.sum(np.abs(w[idx_top])) + 1e-8)
        arrays.append(z_j)
    arrays = np.array(arrays)  # 요인×문항
    if rotate:
        arrays = varimax(arrays.T).T
    return loadings, eigvals, R, arrays  # arrays: Type×Q

def assign_types(loadings: np.ndarray, emails: list, thr=0.40, sep=0.10):
    K = loadings.shape[1]
    max_idx = loadings.argmax(axis=1)
    max_val = loadings.max(axis=1)
    sorted_vals = np.sort(np.abs(loadings), axis=1)[:, ::-1]
    gap = sorted_vals[:,0] - sorted_vals[:,1]
    assigned = (max_val>=thr) & (gap>=sep)
    return pd.DataFrame({
        "email": emails,
        "Type": [f"Type{int(i)+1}" for i in max_idx],
        "MaxLoading": max_val, "Gap": gap, "Assigned": assigned
    })

def top_bottom_statements(arrays: np.ndarray, topk=TOPK_STATEMENTS):
    tb = []
    for t in range(arrays.shape[0]):
        z = arrays[t]
        top_idx = np.argsort(z)[::-1][:topk]
        bot_idx = np.argsort(z)[:topk]
        tb.append((top_idx, bot_idx, z))
    return tb

# ========================= 공통문항 교차분석(1~4) =========================
def scree_and_parallel(df, n_perm=500, show_plot=True):
    """스크리+병렬분석: 숫자형 열만 사용 + 최소 크기 체크"""
    dfn = df.select_dtypes(include=[np.number]).copy()
    if dfn.shape[0] < 5 or dfn.shape[1] < 5:
        raise ValueError("병렬분석: 응답자/문항이 최소 5×5 이상이어야 합니다.")
    R = person_correlation(dfn)
    eigvals = np.linalg.eigvalsh(R)[::-1]
    p = R.shape[0]
    perm_eigs = np.zeros((n_perm, p))
    for b in range(n_perm):
        X = rng.standard_normal(size=dfn.shape)
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, ddof=1, keepdims=True)+1e-8)
        Rb = np.corrcoef(X)
        perm_eigs[b] = np.linalg.eigvalsh(Rb)[::-1]
    mean_perm = perm_eigs.mean(axis=0)
    k_star = int(np.sum(eigvals > mean_perm))
    fig = None
    if show_plot:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1,p+1), eigvals, marker='o', label='Observed')
        ax.plot(range(1,p+1), mean_perm, marker='x', label='Parallel mean')
        ax.axvline(k_star, color='r', linestyle='--', label=f'k*={k_star}')
        ax.set_xlabel('Factor number'); ax.set_ylabel('Eigenvalue'); ax.set_title('Scree + Parallel')
        ax.legend(); fig.tight_layout()
    return {'eigvals': eigvals, 'parallel_mean': mean_perm, 'k_star': k_star, 'fig': fig}

def pca_loadings_on_items(df, k=5):
    dfn = df.select_dtypes(include=[np.number]).copy()
    X = (dfn - dfn.mean(axis=0))/ (dfn.std(axis=0, ddof=1)+1e-8)
    pca = PCA(n_components=k, random_state=RNG_SEED).fit(X)
    L = pca.components_.T
    for j in range(L.shape[1]): L[:,j] /= (norm(L[:,j])+1e-8)
    return L, pca.explained_variance_ratio_.sum()

def procrustes_congruence(LA, LB):
    R, _ = orthogonal_procrustes(LB, LA)  # LB*R ≈ LA
    LB_aligned = LB @ R
    return np.array([float((LA[:,j] @ LB_aligned[:,j])/(norm(LA[:,j])*norm(LB_aligned[:,j])+1e-8)) for j in range(LA.shape[1])])

def congruence_across_sets(dfA, dfB, dfC, common_ids, k=5):
    A = dfA[common_ids].select_dtypes(include=[np.number]).copy()
    B = dfB[common_ids].select_dtypes(include=[np.number]).copy()
    C = dfC[common_ids].select_dtypes(include=[np.number]).copy()
    LA, varA = pca_loadings_on_items(A, k); LB, varB = pca_loadings_on_items(B, k); LC, varC = pca_loadings_on_items(C, k)
    phi_AB = procrustes_congruence(LA, LB)
    phi_AC = procrustes_congruence(LA, LC)
    phi_BC = procrustes_congruence(LB, LC)
    return {'phi_mean_AB': float(np.mean(phi_AB)),
            'phi_mean_AC': float(np.mean(phi_AC)),
            'phi_mean_BC': float(np.mean(phi_BC)),
            'phi_AB': phi_AB, 'phi_AC': phi_AC, 'phi_BC': phi_BC,
            'explained_var': {'A':varA,'B':varB,'C':varC}}

def q_factor_solution(df, k=5):
    dfn = df.select_dtypes(include=[np.number]).copy()
    X = dfn.to_numpy(dtype=float)
    Xs = (X - X.mean(axis=1, keepdims=True))/ (X.std(axis=1, ddof=1, keepdims=True)+1e-8)
    R = np.corrcoef(Xs)
    pca = PCA(n_components=k, random_state=RNG_SEED).fit(R)
    Lp = pca.components_.T  # persons×k
    Z_items = (dfn - dfn.mean(axis=0)) / (dfn.std(axis=0, ddof=1)+1e-8)
    z_arrays = []
    for j in range(k):
        w = Lp[:,j]
        idx = np.argsort(np.abs(w))[::-1][:max(5, int(0.1*len(w)))]
        z_j = (Z_items.iloc[idx].T @ w[idx])/(np.sum(np.abs(w[idx]))+1e-8)
        z_arrays.append(z_j)
    Z = pd.DataFrame(np.column_stack(z_arrays), index=dfn.columns, columns=[f"F{t+1}" for t in range(k)])
    return Z, Lp

def distinguishing_tests(Z, alpha=0.01, se=0.30):
    items = Z.index; k = Z.shape[1]
    rows=[]
    for itm in items:
        row = Z.loc[itm].values
        for a in range(k):
            for b in range(a+1,k):
                diff = row[a]-row[b]; z = diff/(np.sqrt(2)*se+1e-8)
                p = 2*(1 - zdist.cdf(abs(z)))
                if p < alpha:
                    rows.append((itm, f"F{a+1}", f"F{b+1}", diff, z, p))
    return pd.DataFrame(rows, columns=["item","F_high","F_low","z_diff","z_stat","p"]).sort_values("p")

def humphreys_rule(Lp):
    N = Lp.shape[0]; thr = 2*(1/np.sqrt(N))
    flags = {}
    for j in range(Lp.shape[1]):
        w = np.sort(np.abs(Lp[:,j]))[::-1][:2]
        flags[f"F{j+1}"] = bool(w[0]*w[1] > thr)
    return flags, thr

def bootstrap_factor_stability(df_common, k=5, B=500, phi_threshold=0.80):
    dfn = df_common.select_dtypes(include=[np.number]).copy()
    base_L, _ = pca_loadings_on_items(dfn, k)
    N = dfn.shape[0]; phis=[]
    for b in range(B):
        idx = rng.integers(low=0, high=N, size=N)
        Lb, _ = pca_loadings_on_items(dfn.iloc[idx], k)
        R, _ = orthogonal_procrustes(Lb, base_L); Lba = Lb @ R
        phis.append([float((base_L[:,j]@Lba[:,j])/(norm(base_L[:,j])*norm(Lba[:,j])+1e-8)) for j in range(k)])
    PHI = np.array(phis)
    return {'phi_mean': PHI.mean(axis=0),
            'phi_std': PHI.std(axis=0),
            'stability_rate': (PHI >= phi_threshold).mean(axis=0),
            'B': B, 'phi_threshold': phi_threshold}

# ========================= 업로드 & 탭 =========================
st.sidebar.header("데이터 업로드")
file = st.sidebar.file_uploader("엑셀 업로드 (시트명: PARTA, PARTB, PARTC)", type=["xlsx"])
if file is None:
    st.info("엑셀을 업로드하세요.")
    st.stop()

try:
    parts = load_excel_parts(file.getvalue(), sheet_names=("PARTA","PARTB","PARTC"))
    st.sidebar.success("시트 로딩 완료")
except Exception as e:
    st.sidebar.error(f"엑셀 로딩 오류: {e}")
    st.stop()

tabA, tabB, tabC, tabCross, tabDist, tabBoot = st.tabs(["세트 A","세트 B","세트 C","공통 교차분석","구별진술","부트스트랩"])

def run_set_tab(df_set: pd.DataFrame, title="세트"):
    st.subheader(f"{title} — 사람 요인화(Q) 분석")
    df_set, (Q_COLS, Q_SET) = ensure_q_columns(df_set, q_count=None)
    df_q = df_set[Q_COLS].copy()
    mask = df_q.notna().sum(axis=1) >= int(0.6*len(Q_COLS))
    df_q = df_q[mask]; emails = df_set.loc[mask,"email"].fillna("").astype(str).tolist()

    st.write(f"유효 응답자 수: **{len(df_q)}명** / 문항 수: **{len(Q_COLS)}**")
    if len(df_q) < MIN_N_FOR_ANALYSIS:
        st.warning(f"분석에 최소 {MIN_N_FOR_ANALYSIS}명 필요")
        return

    with st.expander("⚙️ 분석 옵션", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            corr_metric = st.selectbox("상관계수", ["Pearson","Spearman"], index=0, key=f"{title}_corr")
        with colB:
            n_f_override = st.number_input("요인 수(0=자동)", 0, 6, 0, 1, key=f"{title}_nf")
            n_factors = None if n_f_override==0 else int(n_f_override)
        with colC:
            rotate = st.checkbox("Varimax 회전", value=True, key=f"{title}_rot")
        thr = st.slider("유형 배정 임계(최대 적재)", 0.20, 0.70, 0.40, 0.05, key=f"{title}_thr")
        sep = st.slider("1등-2등 격차", 0.00, 0.50, 0.10, 0.05, key=f"{title}_sep")

    try:
        loadings, eigvals, R, arrays = person_q_analysis(pd.concat([df_set[["email"]], df_q], axis=1),
                                                         corr_metric, n_factors, rotate)
        K = loadings.shape[1]
        st.markdown(f"**추출 요인 수: {K}**")
        load_df = pd.DataFrame(loadings, columns=[f"Type{i+1}" for i in range(K)])
        load_df.insert(0, "email", emails)
        st.dataframe(load_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

        assign_df = assign_types(loadings, emails, thr=thr, sep=sep)
        st.markdown("### 참가자 유형 배정")
        st.dataframe(assign_df, use_container_width=True)
        st.write("유형별 인원수:", assign_df[assign_df["Assigned"]].groupby("Type").size().to_dict())

        arrays_df = pd.DataFrame(arrays, columns=Q_COLS, index=[f"Type{i+1}" for i in range(K)])
        st.markdown("### 유형별 factor array (진술 z-프로파일)")
        st.dataframe(arrays_df, use_container_width=True)
        st.download_button("📥 유형별 factor array CSV",
                           data=arrays_df.to_csv().encode("utf-8-sig"),
                           file_name=f"{title}_type_factor_arrays.csv", mime="text/csv")

        st.markdown(f"### 유형별 상/하위 진술 Top {TOPK_STATEMENTS}")
        tb = top_bottom_statements(arrays, topk=TOPK_STATEMENTS)
        for i, (top_idx, bot_idx, z) in enumerate(tb, start=1):
            with st.expander(f"Type{i} 상/하위 진술", expanded=(i==1)):
                st.markdown("**상위(+) 진술**")
                for j in top_idx:
                    st.write(f"- {Q_COLS[j]} (z={z[j]:.2f})")
                st.markdown("**하위(−) 진술**")
                for j in bot_idx:
                    st.write(f"- {Q_COLS[j]} (z={z[j]:.2f})")
    except Exception as e:
        st.error(f"{title} 분석 오류: {e}")

with tabA: run_set_tab(parts["A"], "세트 A")
with tabB: run_set_tab(parts["B"], "세트 B")
with tabC: run_set_tab(parts["C"], "세트 C")

# ---------- 공통 교차분석 ----------
with tabCross:
    st.subheader("공통문항 교차분석 (Scree+Parallel, Procrustes, 설명분산)")
    A_cols = [c for c in parts["A"].columns if c!="email"]
    B_cols = [c for c in parts["B"].columns if c!="email"]
    C_cols = [c for c in parts["C"].columns if c!="email"]
    common_auto = sorted(list(set(A_cols) & set(B_cols) & set(C_cols)))
    common_ids = st.multiselect("공통문항 선택", common_auto, default=common_auto)

    if len(common_ids) < 5:
        st.info("공통문항 5개 이상 선택하세요.")
    else:
        col1, col2, col3 = st.columns(3)
        for col, sid in zip([col1,col2,col3], ["A","B","C"]):
            with col:
                try:
                    res = scree_and_parallel(parts[sid][common_ids], n_perm=300, show_plot=True)
                    st.pyplot(res['fig'])
                    st.caption(f"{sid}: k*={res['k_star']}")
                except Exception as e:
                    st.warning(f"{sid} Scree/Parallel 오류: {e}")

        # 🔧 k_rec 계산 시에도 공통문항만 전달 + 숫자열 보장
        try:
            k_rec = int(np.median([
                scree_and_parallel(
                    parts[s][common_ids].select_dtypes(include=[np.number]),  # 🔒 숫자열만
                    n_perm=300, show_plot=False
                )['k_star']
                for s in ["A","B","C"]
                if parts[s][common_ids].select_dtypes(include=[np.number]).shape[0] >= 5
                   and parts[s][common_ids].select_dtypes(include=[np.number]).shape[1] >= 5
            ]))
            k_rec = max(2, min(6, k_rec))
            cong = congruence_across_sets(parts["A"], parts["B"], parts["C"], common_ids, k=k_rec)
            st.write(f"권고 요인 수 k = **{k_rec}**")
            st.dataframe(pd.DataFrame({"pair":["A-B","A-C","B-C"],
                                       "phi_mean":[cong['phi_mean_AB'], cong['phi_mean_AC'], cong['phi_mean_BC']]}))
            st.dataframe(pd.DataFrame({"phi_AB":cong['phi_AB'],
                                       "phi_AC":cong['phi_AC'],
                                       "phi_BC":cong['phi_BC']}))
            st.dataframe(pd.DataFrame(cong['explained_var'], index=["explained_var"]))
        except Exception as e:
            st.error(f"Procrustes/요인 일치도 계산 오류: {e}")

# ---------- 구별진술 ----------
with tabDist:
    st.subheader("구별진술 & Humphrey’s Rule (공통문항 권장)")
    sid = st.selectbox("세트 선택", ["A","B","C"], index=0)
    cols = [c for c in parts[sid].columns if c!="email"]
    use_common = st.checkbox("공통문항만 사용(교집합)", value=True)
    if use_common:
        common_auto = sorted(list(set([c for c in parts["A"].columns if c!="email"]) &
                                  set([c for c in parts["B"].columns if c!="email"]) &
                                  set([c for c in parts["C"].columns if c!="email"])))
        target_cols = common_auto
    else:
        target_cols = cols

    # 🔒 숫자형 검증
    if parts[sid][target_cols].select_dtypes(include=[np.number]).shape[1] < 5:
        st.warning("숫자형 문항이 5개 미만입니다. 공통문항/헤더/응답 형식을 확인하세요.")
    else:
        k_in = st.number_input("요인 수(0=자동)", 0, 6, 0, 1)
        k_use = None if k_in==0 else int(k_in)
        try:
            Z, Lp = q_factor_solution(parts[sid][target_cols], k=k_use if k_use else 5)
            dist = distinguishing_tests(Z, alpha=0.01, se=0.30)
            flags, thr = humphreys_rule(Lp)
            st.markdown(f"Humphrey’s rule 임계: **{thr:.3f}**")
            st.dataframe(pd.DataFrame({"Factor":list(flags.keys()),
                                       "Humphreys_pass":[int(v) for v in flags.values()]}))
            st.markdown("**z-array (문항×요인)**"); st.dataframe(Z)
            st.markdown("**구별진술 후보(유의)**"); st.dataframe(dist.head(50))
        except Exception as e:
            st.error(f"구별진술 분석 오류: {e}")

# ---------- 부트스트랩 ----------
with tabBoot:
    st.subheader("부트스트랩 안정도(공통문항)")
    common_auto = sorted(list(set([c for c in parts["A"].columns if c!="email"]) &
                              set([c for c in parts["B"].columns if c!="email"]) &
                              set([c for c in parts["C"].columns if c!="email"])))
    common_ids = st.multiselect("공통문항 선택", common_auto, default=common_auto)
    B = st.number_input("부트스트랩 반복 수", 100, 2000, 500, 50)
    phi_thr = st.slider("일치 임계 φ", 0.50, 0.95, 0.80, 0.01)
    sid = st.selectbox("세트 선택", ["A","B","C"], index=0)

    if len(common_ids) < 5:
        st.info("공통문항은 최소 5개 이상 선택해 주세요.")
    elif parts[sid][common_ids].select_dtypes(include=[np.number]).shape[1] < 5:
        st.warning("선택한 공통문항 중 숫자형이 5개 미만입니다.")
    else:
        try:
            res = bootstrap_factor_stability(parts[sid][common_ids], k=5, B=int(B), phi_threshold=float(phi_thr))
            st.dataframe(pd.DataFrame({"phi_mean":res['phi_mean'], "phi_std":res['phi_std'],
                                       "stability_rate(>=phi_thr)":res['stability_rate']},
                                      index=[f"F{i+1}" for i in range(len(res['phi_mean']))]))
            st.caption("stability_rate: 부트스트랩 표본 중 φ≥임계 비율(요인별).")
        except Exception as e:
            st.error(f"부트스트랩 오류: {e}")

st.success("앱 준비 완료 — 좌측에서 엑셀 업로드 후 각 탭을 이용하세요.")
