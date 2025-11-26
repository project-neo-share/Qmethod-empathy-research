# -*- coding: utf-8 -*-
"""
Q-정렬 현장 분석 앱 (Optimized for Stability & Speed)
- 주요 개선: Caching 적용, 예외 처리 강화, 수치 해석 안정성 확보
"""

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.stats import norm as zdist

# ========================= 설정 및 상수 =========================
st.set_page_config(page_title="Q-Method Field Analysis", layout="wide")

EMAIL_COL_CAND = ["email", "Email", "E-mail", "respondent", "id", "ID"]
MIN_N_FOR_ANALYSIS = 10  # 분석 가능 최소 인원 완화 (테스트 용이성)
TOPK_STATEMENTS = 5
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ========================= 유틸리티 함수 =========================

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """모든 열을 숫자로 변환, 오류 발생 시 NaN 처리"""
    return df.apply(pd.to_numeric, errors='coerce')

def _looks_like_qcol(name: str) -> bool:
    """문항 열인지 식별 (메타데이터 컬럼 제외)"""
    name_l = str(name).strip().lower()
    meta_cols = ["email", "respondent", "id", "time", "name", "timestamp", "date"]
    return not any(k in name_l for k in meta_cols)

def common_C35_columns(parts_dict):
    """세트 A/B/C의 공통 열 중 C01~C35 패턴 추출 (대소문자 무시)"""
    # C01 ~ C35, c01 ~ c35 허용
    pat = re.compile(r"^C(0[1-9]|[12][0-9]|3[0-5])$", re.IGNORECASE)
    
    def get_cols(df):
        return {c for c in df.columns if pat.match(str(c).strip())}
    
    try:
        cols_sets = [get_cols(parts_dict[k]) for k in ["A", "B", "C"] if k in parts_dict]
        if not cols_sets: return []
        common = set.intersection(*cols_sets)
        return sorted(list(common))
    except:
        return []

@st.cache_data(show_spinner=False)
def load_excel_parts(file_bytes: bytes, sheet_names=("PARTA", "PARTB", "PARTC")):
    """
    엑셀 로딩 및 전처리 (캐싱 적용됨)
    """
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    parts = {}
    
    # 시트 존재 여부 확인
    available_sheets = xls.sheet_names
    target_sheets = [s for s in sheet_names if s in available_sheets]
    
    if not target_sheets:
        raise ValueError(f"지정된 시트가 없습니다. (발견된 시트: {available_sheets})")

    for sname in target_sheets:
        # 헤더 자동 인식
        raw = pd.read_excel(xls, sheet_name=sname)
        
        # Email 컬럼 찾기
        email_col = next((c for c in raw.columns if str(c).strip() in EMAIL_COL_CAND), None)
        
        # Email 컬럼이 없으면 인덱스를 ID로 사용
        if email_col is None:
            raw["_generated_id"] = [f"ID_{i+1}" for i in range(len(raw))]
            email_col = "_generated_id"
            
        # 문항 데이터 정제
        q_cols = [c for c in raw.columns if c != email_col and _looks_like_qcol(c)]
        num_df = _coerce_numeric(raw[q_cols])
        
        # 유효 데이터 필터링 (값이 3개 이상 있는 문항만)
        valid_cols = [c for c in num_df.columns if num_df[c].notna().sum() >= 3]
        
        df_final = num_df[valid_cols].copy()
        df_final.insert(0, "email", raw[email_col].fillna("Unknown").astype(str))
        
        # Key를 A, B, C로 매핑 (PARTA -> A)
        key = sname.replace("PART", "")
        parts[key] = df_final.reset_index(drop=True)
        
    return parts

def ensure_q_columns(df: pd.DataFrame):
    """email과 숫자형 문항 분리"""
    if df.empty:
        return df, ([], [])
        
    # 첫 컬럼을 email로 가정 (load_excel_parts에서 처리됨)
    email_col = df.columns[0]
    dfn = df.select_dtypes(include=[np.number])
    Q_COLS = list(dfn.columns)
    
    df_out = pd.concat([df[[email_col]], dfn], axis=1)
    return df_out, (Q_COLS, [str(c) for c in Q_COLS])

# ========================= 통계 및 분석 코어 =========================

def standardize_rows(X: np.ndarray):
    """행(사람)별 표준화 (Z-score)"""
    # ddof=1 for sample std
    std = X.std(axis=1, ddof=1, keepdims=True)
    # 표준편차 0인 경우 1로 대체하여 나눗셈 오류 방지
    std[std == 0] = 1.0 
    return (X - X.mean(axis=1, keepdims=True)) / std

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """Varimax Rotation (Numpy Implementation)"""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        # SVD 안정성 확보
        u, s, vh = np.linalg.svd(
            np.dot(Phi.T, (Lambda**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        )
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return np.dot(Phi, R)

@st.cache_data(show_spinner=False)
def calculate_person_correlation(data_values: np.ndarray, metric="Pearson"):
    """
    상관계수 계산 (캐싱을 위해 numpy array 입력 받음)
    Input: (n_persons, n_items)
    """
    if metric.lower().startswith("spear"):
        # Rank 변환 (행 별로)
        data_rank = np.apply_along_axis(lambda v: pd.Series(v).rank(method="average").to_numpy(), 1, data_values)
        data_norm = standardize_rows(data_rank)
    else:
        data_norm = standardize_rows(data_values)
    
    # Q-method: 사람 간의 상관계수 (Rows=Persons)
    # np.corrcoef는 행(row)을 변수로 인식하므로 그대로 사용
    return np.corrcoef(data_norm)

def person_q_analysis(df_q: pd.DataFrame, corr_metric="Pearson", n_factors=None, rotate=True):
    """Q-Methodology Factor Analysis Pipeline"""
    # 데이터 준비
    df_only = df_q.drop(columns=["email"], errors="ignore")
    X = df_only.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    
    # 1. 상관계수 행렬 (R)
    R = calculate_person_correlation(X, metric=corr_metric)
    
    # 2. 고유값 분해 (Eigendecomposition)
    eigvals, eigvecs = np.linalg.eigh(R)
    # 내림차순 정렬
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 3. 요인 수 결정
    if not n_factors or n_factors <= 0:
        n_factors = int(np.sum(eigvals > 1.0)) # Kaiser Rule
        n_factors = max(2, min(7, n_factors))  # Safety bounds
        
    # 4. 적재치 (Loadings) 추출 (Centroid/PCA approach approximation)
    # loadings = eigenvector * sqrt(eigenvalue)
    loadings = eigvecs[:, :n_factors] * np.sqrt(eigvals[:n_factors])
    
    # 5. 회전 (Rotation)
    if rotate:
        loadings = varimax(loadings)
        
    # 6. 요인 점수 (Factor Scores / Arrays) 계산
    # 문항별 Z-score (Standardized items across people? No, Q uses weighted average of pure sorts)
    # 여기서는 근사적으로 가중치 합산 방식 사용
    
    # 문항 표준화 (열 방향)
    item_std = X.std(axis=0, ddof=1)
    item_std[item_std==0] = 1.0
    Z_items = (X - X.mean(axis=0)) / item_std
    
    arrays = []
    for j in range(n_factors):
        w = loadings[:, j]
        # 요인 정의에 기여하는 주요 응답자 가중치 (Flagging logic 간소화)
        # Factor Loading의 제곱을 가중치로 사용하거나, 단순히 Loading을 가중치로 사용
        # Q-method 표준: z_factor = sum(loading * z_person) / sqrt(sum(loading^2)) 
        # 본 코드는 기존 로직(Top respondent weighted avg) 유지하되 안정성 보강
        
        weight_sum = np.sum(np.abs(w)) + 1e-9
        z_j = np.dot(Z_items.T, w) / weight_sum # Simple weighted average
        
        # Z-score normalization of the factor array itself
        z_j_std = z_j.std(ddof=1)
        if z_j_std == 0: z_j_std = 1.0
        z_j = (z_j - z_j.mean()) / z_j_std
        
        arrays.append(z_j)
        
    arrays = np.array(arrays) # (Factors x Items)
    
    return loadings, eigvals, R, arrays

def assign_types(loadings: np.ndarray, emails: list, thr=0.40, sep=0.10):
    """참가자 요인 배정 로직"""
    n_persons, n_factors = loadings.shape
    
    abs_loadings = np.abs(loadings)
    max_idx = abs_loadings.argmax(axis=1)
    max_val = abs_loadings.max(axis=1)
    
    # 2번째로 큰 값 찾기 (Gap 계산용)
    sorted_vals = np.sort(abs_loadings, axis=1)[:, ::-1]
    second_val = sorted_vals[:, 1] if n_factors > 1 else np.zeros(n_persons)
    
    gap = max_val - second_val
    
    # 배정 조건: 최대값이 임계치 이상 AND 차이가 sep 이상
    assigned = (max_val >= thr) & (gap >= sep)
    
    # Type 문자열
    types = [f"Type{i+1}" if assign else "None" for i, assign in zip(max_idx, assigned)]
    
    return pd.DataFrame({
        "email": emails,
        "Type": types,
        "MaxLoading": loadings[np.arange(n_persons), max_idx], # 부호 포함 원래 값
        "AbsMax": max_val,
        "Gap": gap,
        "Assigned": assigned
    })

# ========================= 교차 분석 함수 (Caching) =========================

@st.cache_data(show_spinner=False)
def run_scree_parallel(df_values: np.ndarray, n_perm=300):
    """Scree Plot & Parallel Analysis"""
    n_persons, n_items = df_values.shape
    
    # 관측된 고유값
    R = np.corrcoef(standardize_rows(df_values))
    obs_eigs = np.linalg.eigvalsh(R)[::-1] # 내림차순
    obs_eigs = np.maximum(obs_eigs, 0) # 수치적 오차로 인한 음수 제거
    
    # 무작위 순열/노이즈 시뮬레이션
    perm_eigs = np.zeros((n_perm, n_persons))
    
    for b in range(n_perm):
        # Random Normal Noise approach for Parallel Analysis
        noise = rng.standard_normal(size=(n_persons, n_items))
        R_noise = np.corrcoef(standardize_rows(noise))
        eigs_b = np.linalg.eigvalsh(R_noise)[::-1]
        perm_eigs[b] = eigs_b
        
    mean_perm = perm_eigs.mean(axis=0)
    p95_perm = np.percentile(perm_eigs, 95, axis=0)
    
    # k_star: 관측값이 무작위 평균보다 큰 개수
    k_star = int(np.sum(obs_eigs > mean_perm))
    
    return obs_eigs, mean_perm, p95_perm, k_star

def procrustes_congruence(LA, LB):
    """요인 구조 일치도 (Tucker's Congruence Coefficient after Procrustes)"""
    # Procrustes Rotation: LB를 LA에 맞춤
    R, _ = orthogonal_procrustes(LB, LA)
    LB_aligned = np.dot(LB, R)
    
    # Congruence Coefficient (Cosines)
    phis = []
    for j in range(LA.shape[1]):
        num = np.dot(LA[:, j], LB_aligned[:, j])
        den = norm(LA[:, j]) * norm(LB_aligned[:, j]) + 1e-9
        phis.append(float(num / den))
        
    return np.array(phis)

@st.cache_data(show_spinner=False)
def bootstrap_factor_stability(data_values: np.ndarray, k=5, B=500, phi_threshold=0.80):
    """부트스트랩 안정도 검증 (상당한 연산량 -> 캐싱 필수)"""
    N = data_values.shape[0]
    
    # Base Solution
    pca = PCA(n_components=k, random_state=RNG_SEED)
    base_L = pca.fit_transform(standardize_rows(data_values))
    
    phis = []
    for b in range(B):
        # Resample with replacement
        idx = rng.choice(N, size=N, replace=True)
        sample = data_values[idx]
        
        # 3명 미만이면 PCA 불가
        if len(np.unique(idx)) < 3: continue
            
        pca_b = PCA(n_components=k, random_state=None)
        Lb = pca_b.fit_transform(standardize_rows(sample))
        
        # Procrustes & Congruence
        phi_vals = procrustes_congruence(base_L, Lb)
        phis.append(phi_vals)
        
    if not phis:
        return None
        
    PHI = np.array(phis)
    return {
        'phi_mean': PHI.mean(axis=0),
        'phi_std': PHI.std(axis=0),
        'stability_rate': (PHI >= phi_threshold).mean(axis=0)
    }

# ========================= UI 레이아웃 =========================

st.title("📊 Q-Method Field Analysis")
st.caption("Robust & Optimized Ver. | PART A/B/C Integration")

with st.sidebar:
    st.header("1. Data Upload")
    file = st.file_uploader("Upload Excel (Sheets: PARTA, PARTB, PARTC)", type=["xlsx"])
    
    if file:
        try:
            parts = load_excel_parts(file.getvalue())
            st.success(f"Loaded: {', '.join(parts.keys())}")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.info("파일이 없다면 코드를 확인하거나 예제 파일을 업로드하세요.")
        st.stop()

tab_list = ["Set A", "Set B", "Set C", "Cross-Analysis", "Distinguishing", "Stability"]
tabs = st.tabs(tab_list)

# ---------- 세트별 분석 (A, B, C) ----------
def run_set_tab(tab_obj, df_set, set_name):
    with tab_obj:
        st.markdown(f"### {set_name} Analysis")
        
        df_q, (cols, col_names) = ensure_q_columns(df_set)
        
        # 결측치 많은 응답자 제거 (60% 이상 응답 필수)
        valid_mask = df_q[cols].notna().sum(axis=1) >= (len(cols) * 0.6)
        df_clean = df_q[valid_mask].copy()
        
        n_obs = len(df_clean)
        st.markdown(f"**N = {n_obs}** (Valid Respondents) | Items = {len(cols)}")
        
        if n_obs < MIN_N_FOR_ANALYSIS:
            st.warning(f"데이터가 너무 적습니다. (최소 {MIN_N_FOR_ANALYSIS}명 권장)")
            return

        # 옵션
        c1, c2, c3 = st.columns(3)
        with c1:
            metric = st.selectbox("Correlation", ["Pearson", "Spearman"], key=f"corr_{set_name}")
        with c2:
            n_factors = st.number_input("Factors (0=Auto)", 0, 7, 0, key=f"nf_{set_name}")
        with c3:
            rotate = st.checkbox("Varimax Rotation", True, key=f"rot_{set_name}")
            
        # 분석 실행
        try:
            loadings, eigvals, R, arrays = person_q_analysis(
                df_clean, metric, n_factors, rotate
            )
            
            K = loadings.shape[1]
            factor_cols = [f"F{i+1}" for i in range(K)]
            
            # 1. Loadings Table
            st.markdown("#### Factor Loadings")
            load_df = pd.DataFrame(loadings, columns=factor_cols)
            load_df.insert(0, "email", df_clean["email"].values)
            
            # 스타일링: 높은 적재치 강조
            st.dataframe(
                load_df.style.background_gradient(cmap="Blues", subset=factor_cols, vmin=-1, vmax=1),
                use_container_width=True
            )
            
            # 2. Type Assignment
            assign_df = assign_types(loadings, df_clean["email"].values)
            st.markdown("#### Type Assignment Summary")
            counts = assign_df[assign_df["Assigned"]]["Type"].value_counts().sort_index()
            st.write(counts.to_dict())
            
            with st.expander("See Assignment Details"):
                st.dataframe(assign_df)
                
            # 3. Factor Arrays (Z-scores)
            st.markdown("#### Factor Arrays (Item Z-scores)")
            arrays_df = pd.DataFrame(arrays.T, index=col_names, columns=factor_cols)
            st.dataframe(arrays_df.style.background_gradient(cmap="RdBu_r", vmin=-2, vmax=2), use_container_width=True)
            
            # Download
            csv = arrays_df.to_csv().encode('utf-8-sig')
            st.download_button("📥 Download Factor Arrays", csv, f"{set_name}_arrays.csv", "text/csv")
            
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

# 탭 실행
if "A" in parts: run_set_tab(tabs[0], parts["A"], "Set A")
else: tabs[0].info("No data for Set A")

if "B" in parts: run_set_tab(tabs[1], parts["B"], "Set B")
else: tabs[1].info("No data for Set B")

if "C" in parts: run_set_tab(tabs[2], parts["C"], "Set C")
else: tabs[2].info("No data for Set C")


# ---------- 공통 교차분석 ----------
with tabs[3]:
    st.header("Cross-Set Analysis (Common Items)")
    
    common_ids = common_C35_columns(parts)
    selected_common = st.multiselect("Select Common Items (C01~C35)", common_ids, default=common_ids)
    
    if len(selected_common) < 5:
        st.warning("최소 5개 이상의 공통 문항이 필요합니다.")
    else:
        col_res = st.columns(3)
        k_stars = []
        
        # 1. Scree Plots
        for idx, sname in enumerate(["A", "B", "C"]):
            if sname not in parts: continue
            
            with col_res[idx]:
                st.subheader(f"Set {sname}")
                df_part = parts[sname]
                # 공통 문항 데이터 추출 (숫자형)
                data_vals = df_part[selected_common].apply(pd.to_numeric, errors='coerce').fillna(0).values
                
                if data_vals.shape[0] < 5:
                    st.info("Not enough data")
                    continue
                    
                obs, mean_p, _, k_star = run_scree_parallel(data_vals)
                k_stars.append(k_star)
                
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(range(1, len(obs)+1), obs, 'o-', label='Observed')
                ax.plot(range(1, len(mean_p)+1), mean_p, 'x--', label='Simulated', alpha=0.7)
                ax.axvline(k_star, color='red', linestyle=':', alpha=0.5)
                ax.set_title(f"Scree Plot (k*={k_star})")
                ax.legend(fontsize='small')
                st.pyplot(fig)
        
        # 2. Congruence
        if len(k_stars) >= 2:
            st.divider()
            rec_k = int(np.median(k_stars)) if k_stars else 3
            rec_k = max(2, min(5, rec_k))
            
            st.markdown(f"#### Factor Congruence (Recommended k={rec_k})")
            
            # 각 세트의 PCA Loadings 계산
            loadings_map = {}
            for sname in ["A", "B", "C"]:
                if sname in parts:
                    data = parts[sname][selected_common].apply(pd.to_numeric).fillna(0).values
                    # Item Loadings (Variables=Items for consistency check)
                    pca = PCA(n_components=rec_k)
                    # For item congruence, we usually look at item loadings
                    # Normalizing columns (items) standard approach
                    L = pca.fit_transform(data.T) # (Items x k)
                    loadings_map[sname] = L
            
            res_rows = []
            pairs = [('A','B'), ('A','C'), ('B','C')]
            for s1, s2 in pairs:
                if s1 in loadings_map and s2 in loadings_map:
                    phis = procrustes_congruence(loadings_map[s1], loadings_map[s2])
                    res_rows.append({
                        "Pair": f"{s1}-{s2}",
                        "Mean Phi": np.mean(phis),
                        "Min Phi": np.min(phis),
                        "Phi per Factor": [round(p,3) for p in phis]
                    })
            
            st.dataframe(pd.DataFrame(res_rows))

# ---------- 구별 진술 분석 ----------
with tabs[4]:
    st.header("Distinguishing Statements")
    
    target_set = st.selectbox("Target Set", ["A", "B", "C"])
    use_common_only = st.checkbox("Use Common Items Only (C01~C35)", value=True)
    
    if target_set in parts:
        df_curr = parts[target_set]
        
        if use_common_only:
            cols = common_C35_columns(parts)
        else:
            _, (cols, _) = ensure_q_columns(df_curr)
            
        if not cols:
            st.error("No columns found.")
        else:
            data = df_curr[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Quick Factor Analysis for Arrays
            k_dist = st.number_input("Number of Factors", 2, 6, 3, key="dist_k")
            
            # Logic:
            # 1. Calculate Factor Arrays (Z-scores)
            # 2. Calculate Difference between Factors
            # 3. Significance Test
            
            try:
                # Reuse core logic manually for custom columns
                # Person Correlation
                R = np.corrcoef(standardize_rows(data.values))
                eigvals, eigvecs = np.linalg.eigh(R)
                idx = eigvals.argsort()[::-1]
                L = eigvecs[:, idx][:, :k_dist] * np.sqrt(eigvals[idx][:k_dist])
                L = varimax(L)
                
                # Arrays
                Z_items = (data - data.mean()) / (data.std() + 1e-9)
                arrays_list = []
                for j in range(k_dist):
                    w = L[:, j]
                    z_j = np.dot(Z_items.T, w) / (np.sum(np.abs(w)) + 1e-9)
                    arrays_list.append(z_j)
                
                Z_arr = pd.DataFrame(np.array(arrays_list).T, index=cols, columns=[f"F{i+1}" for i in range(k_dist)])
                
                st.subheader("Significant Differences (p < .01)")
                
                diff_rows = []
                SE = 1.0 / np.sqrt(len(data)) # Standard Error approx
                
                for item in Z_arr.index:
                    vals = Z_arr.loc[item].values
                    for i in range(k_dist):
                        for j in range(i+1, k_dist):
                            diff = vals[i] - vals[j]
                            z_score = diff / (np.sqrt(2) * SE * 1.96) # Simplified Z
                            # Simple logic: if difference > threshold
                            # Precise Q-method: diff > 2.58 * (SE * sqrt(2)) for p<.01
                            crit_val = 2.58 * (SE * np.sqrt(2))
                            
                            if abs(diff) > crit_val:
                                diff_rows.append({
                                    "Item": item,
                                    "F_High": f"F{i+1}" if diff > 0 else f"F{j+1}",
                                    "F_Low": f"F{j+1}" if diff > 0 else f"F{i+1}",
                                    "Diff": abs(diff),
                                    "Critical_Val_01": crit_val
                                })
                                
                if diff_rows:
                    st.dataframe(pd.DataFrame(diff_rows).sort_values("Diff", ascending=False), use_container_width=True)
                else:
                    st.info("No distinguishing statements found at p < .01")
                    
            except Exception as e:
                st.error(f"Calculation Error: {e}")

# ---------- 안정도 (Bootstrap) ----------
with tabs[5]:
    st.header("Bootstrap Stability")
    st.caption("Resampling Items/Persons to check factor stability")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        bs_set = st.selectbox("Set for Bootstrap", ["A", "B", "C"])
    with col_b2:
        n_iter = st.number_input("Iterations (B)", 100, 1000, 200, step=100)
        
    if bs_set in parts:
        cols = common_C35_columns(parts)
        if len(cols) < 5:
            st.error("Need more common items.")
        else:
            data_bs = parts[bs_set][cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            if st.button("Run Bootstrap"):
                with st.spinner("Running Bootstrap..."):
                    res = bootstrap_factor_stability(data_bs, k=3, B=n_iter)
                    
                if res:
                    st.markdown("### Stability Results")
                    df_res = pd.DataFrame({
                        "Mean Phi": res['phi_mean'],
                        "Std Phi": res['phi_std'],
                        "Stable Rate (>0.80)": res['stability_rate']
                    }, index=[f"F{i+1}" for i in range(len(res['phi_mean']))])
                    
                    st.dataframe(df_res.style.background_gradient(cmap="Greens", subset=["Stable Rate (>0.80)"]))
                    
                    st.info("해석: Stable Rate가 0.8 이상인 요인은 표본 변동에도 견고한 요인으로 간주됩니다.")
                else:
                    st.error("Bootstrap Failed (Data issue)")
