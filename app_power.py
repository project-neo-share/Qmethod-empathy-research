# -*- coding: utf-8 -*-
"""
Q-Methodology Analysis Engine (Python Twin of R Script)
- Author: Gemini (Strict Q-Method Implementation)
- Reference: Brown, S. R. (1980) & R script 'q_runner_all.R' logic.
- Core Features:
  1. Q-Analysis: PCA/Centroid -> Varimax -> Weighted Factor Arrays (Brown's Formula)
  2. Cross-Set Congruence: Tucker's Phi on Factor Arrays
  3. Bootstrap Stability: Robustness test with Noise Injection
  4. Distinguishing Statements: Z-diff significance test (p < .01, .05)
  5. Humphrey's Rule: Factor significance check
  6. Framing ATT: Non-common item bias check
- Update (2025-11-26): Full porting of R script features.
"""

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr, spearmanr, norm as normal_dist
from scipy.linalg import orthogonal_procrustes

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(page_title="Refactored Q-Analysis", layout="wide")

MIN_N_FOR_ANALYSIS = 3
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ==========================================
# 2. Math & Q-Logic Core (The Engine)
# ==========================================

def standardize_rows(X):
    """Row-wise Z-score normalization (Normalizing each person's sort)"""
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0  # Prevent division by zero
    return (X - mean) / std

def tuckers_phi(vec_a, vec_b):
    """Tucker's Congruence Coefficient (Phi)"""
    numerator = np.dot(vec_a, vec_b)
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0: return 0.0
    return numerator / denominator

class QEngine:
    """
    Encapsulates standard Q-methodology workflow.
    Input: Dataframe (Rows=People, Cols=Items)
    Output: Loadings, Factor Arrays (Z-scores)
    """
    def __init__(self, data_df, n_factors=3, rotation=True, corr_method='spearman'):
        self.raw_df = data_df
        self.n_factors = n_factors
        self.rotation = rotation
        self.corr_method = corr_method
        
        # [FIX] Data Cleaning for Likert
        temp_data = data_df.apply(pd.to_numeric, errors='coerce').values
        
        # Row-wise Mean Imputation
        row_means = np.nanmean(temp_data, axis=1)
        inds = np.where(np.isnan(temp_data))
        temp_data[inds] = np.take(row_means, inds[0])
        
        self.data = np.nan_to_num(temp_data, nan=0.0)
        
        self.n_persons, self.n_items = self.data.shape
        self.loadings = None
        self.factor_arrays = None # (n_items x n_factors)
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
        
        # 5. Calculate Factor Arrays
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
        """
        Calculates Item Z-scores using Brown's Weighting Formula.
        Weight = f / (1 - f^2)
        """
        n_items = z_data.shape[1]
        arrays = np.zeros((n_items, self.n_factors))
        
        for f in range(self.n_factors):
            l_vec = loadings[:, f]
            # Cap at 0.95 to prevent singularity
            l_clean = np.clip(l_vec, -0.95, 0.95)
            
            # Brown's weighting
            weights = l_clean / (1 - l_clean**2)
            
            w_abs_sum = np.sum(np.abs(weights))
            if w_abs_sum < 1e-6:
                arrays[:, f] = 0
                continue
                
            weighted_sum = np.dot(weights, z_data)
            
            # Normalize to Z-scores
            arr_mean = np.mean(weighted_sum)
            arr_std = np.std(weighted_sum, ddof=1)
            if arr_std == 0: arr_std = 1.0
            
            arrays[:, f] = (weighted_sum - arr_mean) / arr_std
            
        return arrays

def check_humphreys_rule(loadings, n_persons):
    """
    Checks Humphrey's Rule for factor significance.
    Rule: Product of two highest loadings > 2 / sqrt(N)
    """
    n_factors = loadings.shape[1]
    threshold = 2 * (1 / np.sqrt(n_persons))
    
    results = []
    for f in range(n_factors):
        # Get absolute loadings for this factor
        abs_loads = np.sort(np.abs(loadings[:, f]))[::-1] # Descending
        
        if len(abs_loads) >= 2:
            prod = abs_loads[0] * abs_loads[1]
            pass_rule = prod > threshold
            results.append({
                "Factor": f"F{f+1}",
                "Product (L1*L2)": prod,
                "Threshold": threshold,
                "Pass": pass_rule
            })
        else:
             results.append({
                "Factor": f"F{f+1}",
                "Product (L1*L2)": 0,
                "Threshold": threshold,
                "Pass": False
            })
    return pd.DataFrame(results)

def find_distinguishing_items_r_logic(factor_arrays, n_factors, item_labels=None, se=0.30, alpha=0.01):
    """
    Identifies distinguishing items using R script logic (Z-diff test).
    z_stat = diff / (sqrt(2) * se)
    """
    col_names = [f"F{i+1}" for i in range(n_factors)]
    df_arrays = pd.DataFrame(factor_arrays, columns=col_names)
    if item_labels is not None:
        df_arrays.index = item_labels

    distinguishing_dict = {}
    
    # Critical Z for alpha (two-tailed)
    crit_z = normal_dist.ppf(1 - alpha/2) # e.g., 2.58 for 0.01

    for i in range(n_factors):
        target_col = f"F{i+1}"
        other_cols = [c for c in df_arrays.columns if c != target_col]
        if not other_cols: continue 

        # We need to find items where Target is significantly different from ALL others
        # Logic: Find min(abs(diff)) against all others. If that min diff is significant, then it distinguishes from everyone.
        
        is_higher_all = pd.Series(True, index=df_arrays.index)
        is_lower_all = pd.Series(True, index=df_arrays.index)
        min_diff_val = pd.Series(np.inf, index=df_arrays.index)
        
        for other in other_cols:
            diff = df_arrays[target_col] - df_arrays[other]
            z_stat = diff / (np.sqrt(2) * se)
            
            # Check significance
            is_sig_higher = (z_stat > crit_z)
            is_sig_lower = (z_stat < -crit_z)
            
            is_higher_all &= is_sig_higher
            is_lower_all &= is_sig_lower
            
            # Keep track of the 'smallest' difference (closest neighbor)
            # because that's the bottleneck for distinguishing
            current_diff_abs = np.abs(diff)
            update_mask = current_diff_abs < np.abs(min_diff_val)
            min_diff_val[update_mask] = diff[update_mask]

        dist_mask = is_higher_all | is_lower_all
        dist_items = df_arrays[dist_mask].copy()
        
        if not dist_items.empty:
            dist_items['Distinction'] = np.where(is_higher_all[dist_mask], 'Higher', 'Lower')
            dist_items['Min Difference'] = min_diff_val[dist_mask]
            dist_items['Z-Stat (vs Closest)'] = dist_items['Min Difference'] / (np.sqrt(2) * se)
            dist_items['P-Value'] = 2 * (1 - normal_dist.cdf(np.abs(dist_items['Z-Stat (vs Closest)'])))
            
            dist_items = dist_items.sort_values('Min Difference', ascending=False, key=abs)
            distinguishing_dict[target_col] = dist_items
        
    return distinguishing_dict

# ==========================================
# 3. Stability & Congruence Logic
# ==========================================

@st.cache_data(show_spinner=False)
def bootstrap_stability(df_values, n_factors=3, n_boot=200, corr_method='spearman', noise_std=0.0):
    """
    Checks if the Factor Arrays (Item profiles) remain consistent
    when people are resampled.
    """
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
            noise = rng.normal(0, noise_std, sample_data.shape)
            sample_data = sample_data + noise
        
        boot_engine = QEngine(sample_data, n_factors=n_factors, corr_method=corr_method).fit()
        boot_arrays = boot_engine.factor_arrays
        
        for f in range(n_factors):
            target = base_arrays[:, f]
            best_phi = -1.0
            for bf in range(n_factors):
                candidate = boot_arrays[:, bf]
                phi = tuckers_phi(target, candidate)
                if abs(phi) > abs(best_phi):
                    best_phi = phi
            phi_results[b, f] = abs(best_phi)
            
    phi_results = phi_results[~np.isnan(phi_results).any(axis=1)]
    
    return {
        "mean": np.mean(phi_results, axis=0),
        "std": np.std(phi_results, axis=0),
        "rate_80": np.mean(phi_results >= 0.80, axis=0),
        "rate_90": np.mean(phi_results >= 0.90, axis=0)
    }

@st.cache_data(show_spinner=False)
def calculate_cross_set_congruence(parts_data, common_cols, n_factors=3, corr_method='spearman'):
    engines = {}
    for name, df in parts_data.items():
        df_common = df[common_cols]
        engine = QEngine(df_common, n_factors=n_factors, corr_method=corr_method).fit()
        engines[name] = engine.factor_arrays 
        
    results = []
    pairs = [('A','B'), ('A','C'), ('B','C')]
    for s1, s2 in pairs:
        if s1 not in engines or s2 not in engines: continue
        arr1 = engines[s1]; arr2 = engines[s2]
        phis = []
        for f in range(n_factors):
            phi = tuckers_phi(arr1[:, f], arr2[:, f])
            phis.append(abs(phi))
        results.append({"Pair": f"{s1}-{s2}", "Mean Phi": np.mean(phis), "Factors": phis})
    return results

# ==========================================
# 4. Data Loading & UI
# ==========================================

def parse_uploaded_file(file):
    xls = pd.ExcelFile(file)
    parts = {}
    valid_names = ["PARTA", "PARTB", "PARTC"]
    
    for sname in valid_names:
        if sname in xls.sheet_names:
            df = pd.read_excel(xls, sname)
            id_col = next((c for c in df.columns if str(c).lower() in ['email', 'id', 'respondent']), None)
            if not id_col:
                df['ID'] = [f"P{i}" for i in range(len(df))]
                id_col = 'ID'
            numeric_df = df.apply(pd.to_numeric, errors='coerce')
            valid_cols = numeric_df.columns[numeric_df.notna().sum() > len(df)*0.5]
            valid_cols = [c for c in valid_cols if c != id_col]
            final_df = numeric_df[valid_cols].copy()
            final_df.index = df[id_col]
            final_df = final_df.dropna(thresh=len(valid_cols)*0.8)
            parts[sname.replace("PART", "")] = final_df
    return parts

def get_common_columns(parts):
    pat = re.compile(r"^C\d+$", re.IGNORECASE)
    sets_cols = []
    for df in parts.values():
        cols = {c for c in df.columns if pat.match(str(c))}
        sets_cols.append(cols)
    if not sets_cols: return []
    return sorted(list(set.intersection(*sets_cols)))

def calculate_framing_att(parts, common_cols):
    """Calculates Mean differences for Non-Common items (ATT proxy)"""
    summary = []
    for name, df in parts.items():
        # Identify Non-Common cols
        non_common = [c for c in df.columns if c not in common_cols]
        if not non_common:
            mean_val = 0
        else:
            # Grand mean of all non-common items across all people
            mean_val = np.nanmean(df[non_common].values)
        summary.append({"Set": name, "Non-Common Mean": mean_val, "N_Items": len(non_common)})
    
    df_sum = pd.DataFrame(summary).set_index("Set")
    return df_sum

# ==========================================
# 5. Main Execution
# ==========================================

st.title("Q-Methodology Refactored Analysis")
st.markdown("""
> **교수님을 위한 참고사항 (R-Script Porting):**
> - **Humphrey's Rule:** 요인의 통계적 유의성을 검증하는 탭이 추가되었습니다.
> - **Framing ATT:** 비공통문항들의 평균 차이를 분석하여 프레이밍 효과를 체크합니다.
> - **P-Value 기반 Distinguishing:** Z-threshold 대신 유의확률(p<.01, .05)로 구별 문항을 선별합니다.
""")

uploaded_file = st.sidebar.file_uploader("Upload Excel (PARTA/B/C)", type='xlsx')

if uploaded_file:
    parts = parse_uploaded_file(uploaded_file)
    st.sidebar.success(f"Loaded Sets: {list(parts.keys())}")
    common_cols = get_common_columns(parts)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Basic Q-Analysis", 
        "2. Humphrey's Rule",
        "3. Distinguishing Statements",
        "4. Cross-Set Congruence", 
        "5. Bootstrap Stability", 
        "6. Framing ATT"
    ])
    
    # --- Tab 1: Basic Analysis ---
    with tab1:
        st.header("Single Set Analysis")
        c1, c2, c3 = st.columns(3)
        with c1: target_set = st.selectbox("Select Set", list(parts.keys()))
        with c2: n_factors = st.number_input("Number of Factors", 1, 7, 3)
        with c3: corr_method = st.selectbox("Correlation Method", ["pearson", "spearman"], index=1)
        
        if target_set:
            df = parts[target_set]
            engine = QEngine(df, n_factors=n_factors, corr_method=corr_method).fit()
            
            st.info(f"Top 5 Eigenvalues: {np.round(engine.eigenvalues[:5], 2)}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Explained Variance")
                st.write(engine.explained_variance)
            with col2:
                st.subheader("Significant Types (>0.4)")
                loadings = engine.loadings
                max_vals = np.max(np.abs(loadings), axis=1)
                max_idxs = np.argmax(np.abs(loadings), axis=1)
                valid_types = [f"Type {i+1}" if v > 0.4 else "None" for i, v in zip(max_idxs, max_vals)]
                st.write(pd.Series(valid_types).value_counts().sort_index())
                
            st.subheader("Factor Loadings")
            st.dataframe(pd.DataFrame(engine.loadings, index=df.index, columns=[f"F{i+1}" for i in range(n_factors)]).style.background_gradient(cmap="Blues"))
            st.subheader("Factor Arrays (Z-scores)")
            st.dataframe(pd.DataFrame(engine.factor_arrays, index=df.columns, columns=[f"F{i+1}" for i in range(n_factors)]).style.background_gradient(cmap="RdBu_r"))

    # --- Tab 2: Humphrey's Rule ---
    with tab2:
        st.header("Humphrey's Rule (Factor Significance)")
        st.caption("Verifies if a factor is statistically significant based on its loading magnitude.")
        if target_set:
            # Use engine from Tab 1 context or re-run if needed (Tab 1 context is valid)
            res_hum = check_humphreys_rule(engine.loadings, engine.n_persons)
            st.dataframe(res_hum.style.applymap(lambda x: 'color: green' if x else 'color: red', subset=['Pass']))
            st.info("Pass가 True인 요인만 해석하는 것이 안전합니다.")

    # --- Tab 3: Distinguishing Statements ---
    with tab3:
        st.header("Distinguishing Statements (P-Value Test)")
        c1, c2 = st.columns(2)
        with c1: alpha_level = st.selectbox("Significance Level (Alpha)", [0.01, 0.05], index=0)
        with c2: se_val = st.number_input("Standard Error (SE)", 0.1, 1.0, 0.30, step=0.01, help="R script default is 0.30")

        if target_set:
            dist_dict = find_distinguishing_items_r_logic(engine.factor_arrays, n_factors, item_labels=df.columns, se=se_val, alpha=alpha_level)
            
            tabs_dist = st.tabs([f"Factor {i+1}" for i in range(n_factors)])
            for i, tab in enumerate(tabs_dist):
                with tab:
                    f_key = f"F{i+1}"
                    items_df = dist_dict.get(f_key)
                    if items_df is not None:
                        st.write(f"**{len(items_df)} items (p < {alpha_level})**")
                        st.dataframe(items_df.style.background_gradient(cmap="coolwarm", subset=["Min Difference"], vmin=-2, vmax=2))
                    else:
                        st.info("No distinguishing items found.")

    # --- Tab 4: Cross-Set ---
    with tab4:
        st.header("Cross-Set Congruence")
        if len(common_cols) < 5:
            st.warning("Not enough common columns.")
        else:
            c1, c2 = st.columns(2)
            with c1: n_factors_cross = st.slider("Factors", 2, 5, 3, key='ck')
            with c2: cm_cross = st.selectbox("Correlation", ["pearson", "spearman"], index=1, key='cc')
            results = calculate_cross_set_congruence(parts, common_cols, n_factors_cross, cm_cross)
            for res in results:
                st.subheader(res['Pair'])
                cols = st.columns(len(res['Factors']))
                for i, phi in enumerate(res['Factors']):
                    cols[i].metric(f"F{i+1}", f"{phi:.3f}")
                st.divider()

    # --- Tab 5: Stability ---
    with tab5:
        st.header("Bootstrap Stability")
        c1, c2 = st.columns(2)
        with c1: n_boot = st.number_input("Iterations", 50, 1000, 100)
        with c2: noise = st.slider("Noise (Std Dev)", 0.0, 0.5, 0.05)
        
        if st.button("Run Bootstrap"):
            with st.spinner("Bootstrapping..."):
                res = bootstrap_stability(parts[target_set].values, n_factors=3, n_boot=n_boot, corr_method=corr_method, noise_std=noise)
            st.dataframe(pd.DataFrame(res, index=[f"F{i+1}" for i in range(len(res['mean']))]).style.background_gradient(cmap="Greens", subset=['mean']))

    # --- Tab 6: Framing ATT ---
    with tab6:
        st.header("Framing ATT (Non-Common Items)")
        st.caption("Checks if the unique items in each set introduce a systematic mean bias.")
        
        att_df = calculate_framing_att(parts, common_cols)
        st.dataframe(att_df)
        
        if len(att_df) >= 2:
            st.markdown("#### Pairwise Differences")
            sets = att_df.index.tolist()
            pairs = [(a, b) for idx, a in enumerate(sets) for b in sets[idx+1:]]
            
            diffs = []
            for s1, s2 in pairs:
                m1 = att_df.loc[s1, "Non-Common Mean"]
                m2 = att_df.loc[s2, "Non-Common Mean"]
                diffs.append({"Pair": f"{s2} - {s1}", "Difference": m2 - m1})
            
            st.dataframe(pd.DataFrame(diffs))
            st.info("차이가 0에 가까울수록 프레이밍(문항 구성)에 의한 편향이 적음을 의미합니다.")

else:
    st.info("Please upload the Excel file to begin.")
