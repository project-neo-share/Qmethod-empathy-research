# -*- coding: utf-8 -*-
"""
Q-Methodology Analysis Engine (Python Twin of R Script)
- Author: Prof. Dr. Songhee Kang
- Reference: Brown, S. R. (1980) & R script 'q_runner_all.R' logic.
- Core Features:
  1. Q-Analysis: PCA/Centroid -> Varimax -> Weighted Factor Arrays (Brown's Formula)
  2. Cross-Set Congruence: Tucker's Phi on Factor Arrays
  3. Bootstrap Stability: Robustness test with Noise Injection
  4. Distinguishing Statements: Z-diff significance test (p < .01, .05)
  5. Humphrey's Rule: Factor significance check
  6. Framing ATT: Non-common item bias check
  7. Demographic Analysis: Factor distribution by user attributes
- Update (2025-11-26): Enforced C01-C35 as common items.
- Update (2025-11-29): Added Demographics Tab & Scree Plot. Korean Interpretations added.
"""

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import pearsonr, spearmanr, norm as normal_dist
from scipy.linalg import orthogonal_procrustes

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(page_title="Refactored Q-Analysis", layout="wide")

MIN_N_FOR_ANALYSIS = 3
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# Demographic Value Mappings
DEMO_MAP = {
    "성별": {1: "남자", 2: "여자"},
    "연령": {1: "18~29세", 2: "30~39세", 3: "40~49세", 4: "50~59세", 5: "60세이상"},
    "거주지역": {1: "서울", 2: "인천/경기", 3: "대전/충청/세종", 4: "광주/전라", 5: "대구/경북", 6: "부산/울산/경남", 7: "강원/제주"},
    "거주기간": {1: "2년 미만", 2: "2~5년", 3: "5~10년", 4: "10년 이상"},
    "학력": {1: "중졸이하", 2: "고졸", 3: "전문대재학이상"},
    "직업": {1: "농/임/어업", 2: "자영업", 3: "판매/서비스직", 4: "생산/기능/노무", 5: "사무/관리/전문", 6: "주부", 7: "학생", 8: "무직/기타"},
    "종사자여부": {1: "있음", 2: "없음", 3: "모름"}, # 가족, 지인의 전력산업 종사자 여부
    "가구소득": {1: "200만원미만", 2: "200~400만원", 3: "400~600만원", 4: "600~800만원", 5: "800만원 이상", 6: "모름/무응답"},
    "개인소득": {1: "200만원미만", 2: "200~400만원", 3: "400~600만원", 4: "600~800만원", 5: "800만원 이상", 6: "모름/무응답"},
    "이념성향": {1: "진보", 2: "중도", 3: "보수"}
}

# Helper to find matching key in DEMO_MAP for a given column name
def find_demo_key(col_name):
    for key in DEMO_MAP.keys():
        # Simple keyword matching
        if key in col_name or (key == "거주기간" and "거주" in col_name and "기간" in col_name):
            return key
        if "종사자" in col_name and key == "종사자여부": return key
        if "가구" in col_name and "소득" in col_name and key == "가구소득": return key
        if "개인" in col_name and "소득" in col_name and key == "개인소득": return key
    return None

# ==========================================
# 2. Math & Q-Logic Core (The Engine)
# ==========================================

def standardize_rows(X):
    """Row-wise Z-score normalization"""
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std

def tuckers_phi(vec_a, vec_b):
    """Tucker's Congruence Coefficient (Phi)"""
    numerator = np.dot(vec_a, vec_b)
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0: return 0.0
    return numerator / denominator

class QEngine:
    def __init__(self, data_df, n_factors=3, rotation=True, corr_method='spearman'):
        self.raw_df = data_df
        self.n_factors = n_factors
        self.rotation = rotation
        self.corr_method = corr_method
        
        # Data Cleaning
        temp_data = data_df.apply(pd.to_numeric, errors='coerce').values
        row_means = np.nanmean(temp_data, axis=1)
        inds = np.where(np.isnan(temp_data))
        temp_data[inds] = np.take(row_means, inds[0])
        
        self.data = np.nan_to_num(temp_data, nan=0.0)
        self.n_persons, self.n_items = self.data.shape
        self.loadings = None
        self.factor_arrays = None
        self.explained_variance = None
        self.eigenvalues = None
        
    def fit(self):
        # 1. Correlation Matrix
        if self.corr_method == 'spearman':
            R, _ = spearmanr(self.data, axis=1)
            z_data = standardize_rows(self.data) 
        else:
            z_data = standardize_rows(self.data)
            R = np.corrcoef(z_data)
        R = np.nan_to_num(R, nan=0.0)
        
        # 2. Eigen Decomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.eigenvalues = eigvals
        
        # 3. Extract Factors
        k = self.n_factors
        valid_eigvals = np.maximum(eigvals[:k], 0)
        L = eigvecs[:, :k] * np.sqrt(valid_eigvals)
        
        # 4. Varimax Rotation
        if self.rotation and k > 1:
            L = self._varimax(L)
            
        self.loadings = L
        self.explained_variance = eigvals[:k]
        
        # 5. Factor Arrays
        self.factor_arrays = self._calculate_factor_arrays(L, z_data)
        return self

    def _varimax(self, Phi, gamma=1.0, q=20, tol=1e-6):
        p, k = Phi.shape
        R = np.eye(k)
        d = 0
        for i in range(q):
            d_old = d
            Lambda = np.dot(Phi, R)
            u, s, vh = np.linalg.svd(
                np.dot(Phi.T, (Lambda**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
            )
            R = np.dot(u, vh)
            d = np.sum(s)
            if d_old != 0 and d/d_old < 1 + tol: break
        return np.dot(Phi, R)

    def _calculate_factor_arrays(self, loadings, z_data):
        n_items = z_data.shape[1]
        arrays = np.zeros((n_items, self.n_factors))
        for f in range(self.n_factors):
            l_vec = loadings[:, f]
            l_clean = np.clip(l_vec, -0.95, 0.95)
            weights = l_clean / (1 - l_clean**2)
            if np.sum(np.abs(weights)) < 1e-6:
                arrays[:, f] = 0
                continue
            weighted_sum = np.dot(weights, z_data)
            arr_mean = np.mean(weighted_sum)
            arr_std = np.std(weighted_sum, ddof=1)
            if arr_std == 0: arr_std = 1.0
            arrays[:, f] = (weighted_sum - arr_mean) / arr_std
        return arrays

def check_humphreys_rule(loadings, n_persons):
    n_factors = loadings.shape[1]
    threshold = 2 * (1 / np.sqrt(n_persons))
    results = []
    for f in range(n_factors):
        abs_loads = np.sort(np.abs(loadings[:, f]))[::-1]
        if len(abs_loads) >= 2:
            prod = abs_loads[0] * abs_loads[1]
            pass_rule = prod > threshold
            results.append({"Factor": f"F{f+1}", "Product": prod, "Threshold": threshold, "Pass": pass_rule})
        else:
             results.append({"Factor": f"F{f+1}", "Product": 0, "Threshold": threshold, "Pass": False})
    return pd.DataFrame(results)

def find_distinguishing_items_r_logic(factor_arrays, n_factors, item_labels=None, se=0.30, alpha=0.01):
    col_names = [f"F{i+1}" for i in range(n_factors)]
    df_arrays = pd.DataFrame(factor_arrays, columns=col_names)
    if item_labels is not None: df_arrays.index = item_labels
    distinguishing_dict = {}
    crit_z = normal_dist.ppf(1 - alpha/2)

    for i in range(n_factors):
        target_col = f"F{i+1}"
        other_cols = [c for c in df_arrays.columns if c != target_col]
        if not other_cols: continue 
        
        is_higher_all = pd.Series(True, index=df_arrays.index)
        is_lower_all = pd.Series(True, index=df_arrays.index)
        min_diff_val = pd.Series(np.inf, index=df_arrays.index)
        
        for other in other_cols:
            diff = df_arrays[target_col] - df_arrays[other]
            z_stat = diff / (np.sqrt(2) * se)
            is_higher_all &= (z_stat > crit_z)
            is_lower_all &= (z_stat < -crit_z)
            
            current_diff_abs = np.abs(diff)
            update_mask = current_diff_abs < np.abs(min_diff_val)
            min_diff_val[update_mask] = diff[update_mask]

        dist_mask = is_higher_all | is_lower_all
        dist_items = df_arrays[dist_mask].copy()
        
        if not dist_items.empty:
            dist_items['Distinction'] = np.where(is_higher_all[dist_mask], 'Higher', 'Lower')
            dist_items['Min Difference'] = min_diff_val[dist_mask]
            dist_items['Z-Stat'] = dist_items['Min Difference'] / (np.sqrt(2) * se)
            dist_items['P-Value'] = 2 * (1 - normal_dist.cdf(np.abs(dist_items['Z-Stat'])))
            dist_items = dist_items.sort_values('Min Difference', ascending=False, key=abs)
            distinguishing_dict[target_col] = dist_items
    return distinguishing_dict

# ==========================================
# 3. Stability & Congruence Logic
# ==========================================

@st.cache_data(show_spinner=False)
def bootstrap_stability(df_values, n_factors=3, n_boot=200, corr_method='spearman', noise_std=0.0):
    base_engine = QEngine(pd.DataFrame(df_values), n_factors=n_factors, corr_method=corr_method).fit()
    base_arrays = base_engine.factor_arrays 
    n_persons = df_values.shape[0]
    phi_results = np.zeros((n_boot, n_factors))
    
    for b in range(n_boot):
        indices = rng.choice(n_persons, size=n_persons, replace=True)
        if len(np.unique(indices)) < 3:
            phi_results[b, :] = np.nan
            continue
        sample_data = pd.DataFrame(df_values[indices])
        if noise_std > 0:
            sample_data += rng.normal(0, noise_std, sample_data.shape)
        
        boot_engine = QEngine(sample_data, n_factors=n_factors, corr_method=corr_method).fit()
        boot_arrays = boot_engine.factor_arrays
        
        for f in range(n_factors):
            target = base_arrays[:, f]
            best_phi = -1.0
            for bf in range(n_factors):
                phi = tuckers_phi(target, boot_arrays[:, bf])
                if abs(phi) > abs(best_phi): best_phi = phi
            phi_results[b, f] = abs(best_phi)
    
    phi_results = phi_results[~np.isnan(phi_results).any(axis=1)]
    return {
        "mean": np.mean(phi_results, axis=0),
        "std": np.std(phi_results, axis=0),
        "rate_90": np.mean(phi_results >= 0.90, axis=0)
    }

@st.cache_data(show_spinner=False)
def calculate_cross_set_congruence(parts_q, common_cols, n_factors=3, corr_method='spearman'):
    engines = {}
    for name, df in parts_q.items():
        df_common = df[common_cols]
        engine = QEngine(df_common, n_factors=n_factors, corr_method=corr_method).fit()
        engines[name] = engine.factor_arrays 
    results = []
    pairs = [('A','B'), ('A','C'), ('B','C')]
    for s1, s2 in pairs:
        if s1 not in engines or s2 not in engines: continue
        arr1, arr2 = engines[s1], engines[s2]
        phis = [abs(tuckers_phi(arr1[:,f], arr2[:,f])) for f in range(n_factors)]
        results.append({"Pair": f"{s1}-{s2}", "Mean Phi": np.mean(phis), "Factors": phis})
    return results

# ==========================================
# 4. Data Loading & UI
# ==========================================

def parse_uploaded_file(file):
    xls = pd.ExcelFile(file)
    valid_names = ["PARTA", "PARTB", "PARTC"]
    
    # Store Q-data and Demo-data separately
    q_parts = {}
    d_parts = {}
    
    # Metadata keywords to exclude
    meta_keywords = ['time', 'date', 'duration', 'ip', 'token']
    # Q-item pattern (C01, C02...)
    q_pattern = re.compile(r"^C(0[1-9]|[12][0-9]|3[0-9]|4[0-9])$", re.IGNORECASE)

    for sname in valid_names:
        if sname in xls.sheet_names:
            df = pd.read_excel(xls, sname)
            # Standardize ID column
            id_col = next((c for c in df.columns if str(c).lower() in ['email', 'id', 'respondent', 'pid']), None)
            if not id_col:
                df['ID'] = [f"P{i+1}" for i in range(len(df))]
                id_col = 'ID'
            else:
                df[id_col] = df[id_col].astype(str)
            
            df = df.set_index(id_col)
            numeric_df = df.apply(pd.to_numeric, errors='coerce')

            # Split Columns
            q_cols = []
            d_cols = []
            
            for c in df.columns:
                if q_pattern.match(str(c)):
                    q_cols.append(c)
                elif c not in [id_col] and not any(k in str(c).lower() for k in meta_keywords):
                    # Potential demographic column if numeric and not metadata
                    # Check if it has numeric content (not all NaNs)
                    if numeric_df[c].notna().sum() > 0:
                        d_cols.append(c)

            # Clean Q-Data
            if q_cols:
                q_df = numeric_df[q_cols].dropna(thresh=len(q_cols)*0.8)
                q_parts[sname.replace("PART", "")] = q_df
                
                # Keep corresponding Demographics for the valid Q-sort rows
                d_df = df.loc[q_df.index, d_cols]
                d_parts[sname.replace("PART", "")] = d_df
                
    return q_parts, d_parts

def get_common_columns(parts):
    # Strict Regex C01-C35
    pat = re.compile(r"^C(0[1-9]|[12][0-9]|3[0-5])$", re.IGNORECASE)
    sets_cols = []
    for df in parts.values():
        cols = {c for c in df.columns if pat.match(str(c))}
        sets_cols.append(cols)
    if not sets_cols: return []
    return sorted(list(set.intersection(*sets_cols)))

def calculate_framing_att(parts, common_cols):
    summary = []
    for name, df in parts.items():
        non_common = [c for c in df.columns if c not in common_cols]
        if not non_common:
            mean_val, n_items, items_str = 0, 0, "-"
        else:
            mean_val = np.nanmean(df[non_common].values)
            n_items = len(non_common)
            items_str = f"{non_common[0]}...{non_common[-1]} ({len(non_common)})" if len(non_common)>5 else ", ".join(non_common)
        summary.append({"Set": name, "Non-Common Mean": mean_val, "N_Items": n_items, "Items": items_str})
    return pd.DataFrame(summary).set_index("Set")

# ==========================================
# 5. Main Execution
# ==========================================

st.title("Q-방법론 분석 엔진 v2.0")
st.markdown("""
> **업데이트 (2025-11-29):**
> 1. **인구통계학적 분석 (Tab 7):** 성별, 연령, 소득 등 주요 변수별 요인 분포 분석 기능 추가.
> 2. **스크리 도표 (Scree Plot):** 고유값 변화 그래프 추가 (Tab 1).
> 3. **한글 해석 강화:** 각 분석 탭에 상세한 한글 설명 박스(Interpretation) 추가.
""")

uploaded_file = st.sidebar.file_uploader("Upload Excel (PARTA/B/C)", type='xlsx')

if uploaded_file:
    q_parts, d_parts = parse_uploaded_file(uploaded_file)
    
    if not q_parts:
        st.error("데이터를 읽을 수 없습니다. 시트명(PARTA...)과 문항 코드(C01...)를 확인해주세요.")
    else:
        st.sidebar.success(f"Loaded Sets: {list(q_parts.keys())}")
        common_cols = get_common_columns(q_parts)
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "1. Basic Q-Analysis", 
            "2. Humphrey's Rule",
            "3. Distinguishing Items",
            "4. Cross-Set Congruence", 
            "5. Bootstrap Stability", 
            "6. Framing ATT",
            "7. Demographic Analysis"
        ])
        
        # Shared State
        target_set = st.sidebar.selectbox("Target Set for Tabs 1,2,3,5,7", list(q_parts.keys()))
        n_factors = st.sidebar.number_input("Number of Factors", 1, 7, 3)
        corr_method = st.sidebar.selectbox("Correlation Method", ["pearson", "spearman"], index=1)
        
        # Run Engine for Target Set
        df = q_parts[target_set]
        engine = QEngine(df, n_factors=n_factors, corr_method=corr_method).fit()

        # --- Tab 1: Basic Analysis ---
        with tab1:
            st.header("Basic Q-Analysis Result")
            st.info("💡 **해석(Interpretation):**\n- **Explained Variance:** 각 요인이 전체 데이터의 변동성을 얼마나 설명하는지 나타냅니다. (보통 40% 이상이면 양호)\n- **Scree Plot:** 고유값이 급격히 떨어지다가 완만해지는 지점(Elbow) 전까지를 의미 있는 요인 수로 봅니다.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Explained Variance")
                st.dataframe(pd.DataFrame(engine.explained_variance, index=[f"F{i+1}" for i in range(n_factors)], columns=["Eigenvalue"]).T)
                
                # Scree Plot
                st.subheader("Scree Plot")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(1, len(engine.eigenvalues)+1), engine.eigenvalues, 'bo-', markersize=6)
                ax.axhline(y=1.0, color='r', linestyle='--', linewidth=0.8, label='Eigenvalue=1')
                ax.set_title("Scree Plot (Eigenvalues)")
                ax.set_xlabel("Factor Number")
                ax.set_ylabel("Eigenvalue")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
            with col2:
                st.subheader("Factor Distribution")
                # Factor Assignment (Highest Loading)
                max_vals = np.max(np.abs(engine.loadings), axis=1)
                max_idxs = np.argmax(np.abs(engine.loadings), axis=1)
                valid_types = [f"Type {i+1}" if v > 0.4 else "None" for i, v in zip(max_idxs, max_vals)]
                
                s_counts = pd.Series(valid_types).value_counts().sort_index()
                st.bar_chart(s_counts)
                st.caption(f"Total Participants: {len(df)}")

            st.subheader("Factor Loadings (Rotated)")
            st.dataframe(pd.DataFrame(engine.loadings, index=df.index, columns=[f"F{i+1}" for i in range(n_factors)]).style.background_gradient(cmap="Blues"))

        # --- Tab 2: Humphrey's Rule ---
        with tab2:
            st.header("Humphrey's Rule Validation")
            st.info("💡 **해석(Interpretation):**\n- 험프리 법칙은 요인의 통계적 유의성을 검증합니다.\n- 두 개의 가장 높은 적재량의 곱이 표준오차의 2배를 넘어야 유의미한 요인으로 간주합니다.\n- 'Pass'가 True인 요인만 해석하는 것이 안전합니다.")
            
            res_hum = check_humphreys_rule(engine.loadings, engine.n_persons)
            st.dataframe(res_hum.style.applymap(lambda x: 'color: green; font-weight: bold' if x else 'color: red', subset=['Pass']))

        # --- Tab 3: Distinguishing Items ---
        with tab3:
            st.header("Distinguishing Statements")
            st.info("💡 **해석(Interpretation):**\n- 특정 요인이 다른 모든 요인과 통계적으로 유의미하게(p<.01 or .05) 차이나는 문항들입니다.\n- 이 문항들은 해당 유형(Type)의 독특한 특성을 설명하는 핵심 단서가 됩니다.")
            
            c1, c2 = st.columns(2)
            with c1: alpha_level = st.selectbox("Alpha Level", [0.01, 0.05], index=0)
            with c2: se_val = st.number_input("Standard Error", 0.1, 1.0, 0.30, step=0.01)

            dist_dict = find_distinguishing_items_r_logic(engine.factor_arrays, n_factors, item_labels=df.columns, se=se_val, alpha=alpha_level)
            
            subtabs = st.tabs([f"Type {i+1}" for i in range(n_factors)])
            for i, stab in enumerate(subtabs):
                with stab:
                    f_key = f"F{i+1}"
                    items_df = dist_dict.get(f_key)
                    if items_df is not None:
                        st.write(f"**Found {len(items_df)} distinguishing items (p < {alpha_level})**")
                        st.dataframe(items_df.style.background_gradient(cmap="coolwarm", subset=["Min Difference"], vmin=-2, vmax=2))
                    else:
                        st.warning("No distinguishing items found for this factor.")

        # --- Tab 4: Cross-Set Congruence ---
        with tab4:
            st.header("Cross-Set Congruence")
            st.info("💡 **해석(Interpretation):**\n- Tucker's Phi 계수를 사용하여 서로 다른 데이터셋(PART A, B, C)에서 도출된 요인들이 얼마나 유사한지 비교합니다.\n- **0.90 이상:** 매우 높은 유사성 (동일 요인으로 간주 가능)\n- **0.80 ~ 0.90:** 높은 유사성")
            
            if len(common_cols) < 5:
                st.warning("공통 문항(C01~C35)이 충분하지 않아 교차 분석을 수행할 수 없습니다.")
            else:
                results = calculate_cross_set_congruence(q_parts, common_cols, n_factors, corr_method)
                for res in results:
                    st.subheader(f"Comparison: {res['Pair']}")
                    cols = st.columns(len(res['Factors']))
                    for i, phi in enumerate(res['Factors']):
                        cols[i].metric(f"Factor {i+1}", f"{phi:.3f}", delta_color="normal" if phi>0.9 else "off")
                    st.divider()

        # --- Tab 5: Stability ---
        with tab5:
            st.header("Bootstrap Stability")
            st.info("💡 **해석(Interpretation):**\n- 데이터를 무작위로 복원 추출(Resampling)하여 요인 구조가 얼마나 안정적인지 테스트합니다.\n- **Rate > 0.90**은 요인 구조가 매우 견고함을 의미합니다.")
            
            if st.button("Run Bootstrap Analysis"):
                with st.spinner("Running 100 iterations..."):
                    res = bootstrap_stability(df.values, n_factors=3, n_boot=100, corr_method=corr_method, noise_std=0.05)
                st.dataframe(pd.DataFrame(res, index=[f"F{i+1}" for i in range(len(res['mean']))]).style.background_gradient(cmap="Greens", subset=['mean']))

        # --- Tab 6: Framing ATT ---
        with tab6:
            st.header("Framing ATT (Non-Common Items)")
            st.info("💡 **해석(Interpretation):**\n- 공통 문항(Common Items)을 제외한 비공통 문항(Unique Items)들의 평균 점수 차이를 분석합니다.\n- 차이가 클수록 문항 구성(Framing)에 따른 응답 편향이 존재할 가능성이 있습니다.")
            
            att_df = calculate_framing_att(q_parts, common_cols)
            st.dataframe(att_df)
            
            if len(att_df) >= 2:
                st.markdown("#### Pairwise Differences")
                sets = att_df.index.tolist()
                pairs = [(a, b) for idx, a in enumerate(sets) for b in sets[idx+1:]]
                diffs = [{"Pair": f"{s2} - {s1}", "Difference": att_df.loc[s2, "Non-Common Mean"] - att_df.loc[s1, "Non-Common Mean"]} for s1, s2 in pairs]
                st.dataframe(pd.DataFrame(diffs))

        # --- Tab 7: Demographic Analysis (NEW) ---
        with tab7:
            st.header("Demographic Analysis")
            st.info("💡 **해석(Interpretation):**\n- 각 요인(Type)에 속한 사람들의 인구통계학적 특성(성별, 연령, 직업 등)을 분석합니다.\n- 제공된 코딩값(1, 2, 3...)은 자동으로 라벨(남자, 여자...)로 변환됩니다.")
            
            # 1. Assign Factors to Persons
            max_idxs = np.argmax(np.abs(engine.loadings), axis=1)
            # Only consider valid if loading > 0.4 (Optional, but usually good practice. Here we classify everyone for demo analysis)
            # If strictly needed, filter: factor_labels = [f"Type {i+1}" if abs(engine.loadings[r, i]) > 0.4 else "None" for r, i in enumerate(max_idxs)]
            factor_labels = [f"Type {i+1}" for i in max_idxs]
            
            # 2. Merge with Demographics
            if target_set in d_parts:
                demo_df = d_parts[target_set].copy()
                # Ensure indices match
                common_indices = demo_df.index.intersection(df.index)
                if len(common_indices) == 0:
                    st.error("Q-sort 데이터와 인구통계 데이터의 ID(PID)가 일치하지 않습니다.")
                else:
                    demo_subset = demo_df.loc[common_indices]
                    # Add Factor Column
                    # Use df.index.get_indexer to map factors correctly
                    factor_series = pd.Series(factor_labels, index=df.index)
                    demo_subset['Assigned_Factor'] = factor_series.loc[common_indices]
                    
                    st.write(f"**Analyzed Participants:** {len(demo_subset)} 명")
                    
                    # 3. Iterate Analysis
                    found_demo_cols = False
                    for col in demo_subset.columns:
                        if col == 'Assigned_Factor': continue
                        
                        # Find mapping key
                        map_key = find_demo_key(str(col))
                        
                        if map_key:
                            found_demo_cols = True
                            st.markdown(f"### {col} ({map_key})")
                            
                            # Apply Mapping
                            mapped_col = demo_subset[col].map(DEMO_MAP[map_key])
                            # Handle unmapped values (keep original or mark unknown)
                            mapped_col = mapped_col.fillna("Unknown/Other")
                            
                            # Cross-tabulation
                            ct = pd.crosstab(mapped_col, demo_subset['Assigned_Factor'])
                            
                            # Display Side-by-Side
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                st.dataframe(ct)
                            with c2:
                                st.bar_chart(ct)
                            st.divider()
                    
                    if not found_demo_cols:
                        st.warning("지정된 인구통계 컬럼(성별, 연령 등)을 찾을 수 없습니다. 엑셀 파일의 컬럼명을 확인해주세요.")
                        st.dataframe(demo_subset.head())
            else:
                st.warning("이 데이터셋(Set)에는 인구통계 정보가 없습니다.")

else:
    st.info("분석할 엑셀 파일(PARTA, PARTB, PARTC 시트 포함)을 업로드해주세요.")
