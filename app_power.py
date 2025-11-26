# -*- coding: utf-8 -*-
"""
Q-Methodology Analysis Engine (Refactored for Accuracy)
- Author: Gemini (Strict Q-Method Implementation)
- Reference: Brown, S. R. (1980). Political subjectivity.
- Key Fix: Comparing 'Factor Arrays' (Item Z-scores) for stability/congruence, not Person Loadings.
- Update (Fix): Capped weights to prevent singularity (bootstrap=1 issue) and improved type assignment logic.
- Update (2025-11-26): Optimized for Likert 7-point scale (Row-mean imputation, Spearman option).
- Update (Features): Added 'Distinguishing Statements' tab and 'Noise Injection' for robust bootstrap.
"""

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr, spearmanr
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
    # Use nanmean/nanstd to handle potential NaNs safely before they are filled/processed
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
    def __init__(self, data_df, n_factors=3, rotation=True, corr_method='pearson'):
        self.raw_df = data_df
        self.n_factors = n_factors
        self.rotation = rotation
        self.corr_method = corr_method
        
        # [FIX] Data Cleaning for Likert
        # 1. Coerce to numeric
        temp_data = data_df.apply(pd.to_numeric, errors='coerce').values
        
        # 2. Row-wise Mean Imputation (Better than 0 for 1-7 Likert)
        # If a user missed a question, assume their average response (neutral for them)
        row_means = np.nanmean(temp_data, axis=1)
        inds = np.where(np.isnan(temp_data))
        temp_data[inds] = np.take(row_means, inds[0])
        
        # 3. Fill remaining NaNs (e.g., if full row was NaN) with 0 or drop? 
        # For now, 0, but rows should have been filtered before.
        self.data = np.nan_to_num(temp_data, nan=0.0)
        
        self.n_persons, self.n_items = self.data.shape
        self.loadings = None
        self.factor_arrays = None # (n_items x n_factors)
        self.explained_variance = None
        self.eigenvalues = None
        
    def fit(self):
        # 1. Correlation Matrix (Person x Person)
        if self.corr_method == 'spearman':
            # Spearman rank correlation
            R, _ = spearmanr(self.data, axis=1)
            # Standardize for factor array calculation later
            z_data = standardize_rows(self.data) 
        else:
            # Pearson on standardized sorts (Standard Q)
            z_data = standardize_rows(self.data)
            R = np.corrcoef(z_data)
        
        # Handle NaN correlations (if standard deviation was 0)
        R = np.nan_to_num(R, nan=0.0)
        
        # 2. Eigen Decomposition (Centroid/PCA approx)
        eigvals, eigvecs = np.linalg.eigh(R)
        # Sort descending
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        self.eigenvalues = eigvals # Store for Scree
        
        # 3. Extract Factors
        # Loadings = Eigenvector * sqrt(Eigenvalue)
        k = self.n_factors
        # Ensure non-negative eigenvalues for sqrt
        valid_eigvals = np.maximum(eigvals[:k], 0)
        L = eigvecs[:, :k] * np.sqrt(valid_eigvals)
        
        # 4. Varimax Rotation
        if self.rotation and k > 1:
            L = self._varimax(L)
            
        self.loadings = L
        self.explained_variance = eigvals[:k]
        
        # 5. Calculate Factor Arrays (The Critical Step)
        # Always use Z-scores for array calculation to normalize scale differences
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
        Calculates Item Z-scores for each factor.
        Uses Standard Q Formula: Weight = f / (1 - f^2)
        Fix: Caps loadings at 0.95 to avoid singularity/dominance.
        """
        n_items = z_data.shape[1]
        arrays = np.zeros((n_items, self.n_factors))
        
        for f in range(self.n_factors):
            l_vec = loadings[:, f]
            
            # [FIX] Cap loadings at 0.95. 
            l_clean = np.clip(l_vec, -0.95, 0.95)
            
            weights = l_clean / (1 - l_clean**2)
            
            # If all weights are near zero (unlikely), handle gracefully
            w_abs_sum = np.sum(np.abs(weights))
            if w_abs_sum < 1e-6:
                arrays[:, f] = 0
                continue
                
            # Weighted sum of sorts
            weighted_sum = np.dot(weights, z_data)
            
            # Normalize to Z-scores (Standardize array)
            arr_mean = np.mean(weighted_sum)
            arr_std = np.std(weighted_sum, ddof=1)
            if arr_std == 0: arr_std = 1.0
            
            arrays[:, f] = (weighted_sum - arr_mean) / arr_std
            
        return arrays # (Items x Factors)

def find_distinguishing_items(factor_arrays, n_factors, threshold=1.0, item_labels=None):
    """
    Identifies items that distinguish each factor from all others.
    """
    col_names = [f"F{i+1}" for i in range(n_factors)]
    df_arrays = pd.DataFrame(factor_arrays, columns=col_names)
    if item_labels is not None:
        df_arrays.index = item_labels

    distinguishing_dict = {}

    for i in range(n_factors):
        target_col = f"F{i+1}"
        other_cols = [c for c in df_arrays.columns if c != target_col]
        
        if not other_cols: continue 

        # 1. Compare target vs max of others (Is it significantly HIGHER?)
        # Z_target > Z_other + threshold (for all others)
        diff_high = df_arrays[target_col] - df_arrays[other_cols].max(axis=1)
        dist_high_mask = diff_high > threshold
        
        # 2. Compare target vs min of others (Is it significantly LOWER?)
        # Z_target < Z_other - threshold (for all others)
        diff_low = df_arrays[target_col] - df_arrays[other_cols].min(axis=1)
        dist_low_mask = diff_low < -threshold
        
        # Combine
        dist_mask = dist_high_mask | dist_low_mask
        dist_items = df_arrays[dist_mask].copy()
        
        # Add metadata
        dist_items['Distinction'] = np.where(dist_high_mask[dist_mask], 'Higher', 'Lower')
        dist_items['Difference'] = np.where(dist_high_mask[dist_mask], diff_high[dist_mask], diff_low[dist_mask])
        dist_items = dist_items.sort_values('Difference', ascending=False, key=abs)
        
        distinguishing_dict[target_col] = dist_items
        
    return distinguishing_dict

# ==========================================
# 3. Stability & Congruence Logic
# ==========================================

@st.cache_data(show_spinner=False)
def bootstrap_stability(df_values, n_factors=3, n_boot=200, corr_method='pearson', noise_std=0.0):
    """
    Checks if the Factor Arrays (Item profiles) remain consistent
    when people are resampled.
    
    [Feature] noise_std: Inject random gaussian noise (mean=0, std=noise_std) 
    to prevent trivial 1.0 stability in small samples or single-person dominance.
    """
    # 1. Original Solution
    base_engine = QEngine(pd.DataFrame(df_values), n_factors=n_factors, corr_method=corr_method).fit()
    base_arrays = base_engine.factor_arrays # (Items x Factors)
    
    n_persons = df_values.shape[0]
    phi_results = np.zeros((n_boot, n_factors))
    
    for b in range(n_boot):
        # Resample People (Rows)
        indices = rng.choice(n_persons, size=n_persons, replace=True)
        # Check if we have enough variance (at least 3 distinct people)
        if len(np.unique(indices)) < 3:
            phi_results[b, :] = np.nan
            continue
            
        sample_data = pd.DataFrame(df_values[indices])
        
        # [Robustness] Inject Noise if requested
        if noise_std > 0:
            noise = rng.normal(0, noise_std, sample_data.shape)
            sample_data = sample_data + noise
        
        # Run Q-Analysis on Sample
        boot_engine = QEngine(sample_data, n_factors=n_factors, corr_method=corr_method).fit()
        boot_arrays = boot_engine.factor_arrays
        
        # Compare Factors (Best Match Strategy)
        for f in range(n_factors):
            target = base_arrays[:, f]
            
            # Find best match in boot_arrays
            best_phi = -1.0
            for bf in range(n_factors):
                candidate = boot_arrays[:, bf]
                phi = tuckers_phi(target, candidate)
                if abs(phi) > abs(best_phi):
                    best_phi = phi
            
            phi_results[b, f] = abs(best_phi)
            
    # Clean NaNs
    phi_results = phi_results[~np.isnan(phi_results).any(axis=1)]
    
    return {
        "mean": np.mean(phi_results, axis=0),
        "std": np.std(phi_results, axis=0),
        "rate_80": np.mean(phi_results >= 0.80, axis=0),
        "rate_90": np.mean(phi_results >= 0.90, axis=0)
    }

@st.cache_data(show_spinner=False)
def calculate_cross_set_congruence(parts_data, common_cols, n_factors=3, corr_method='pearson'):
    """
    Calculates Factor Arrays for Set A, B, C separately,
    then calculates Phi between them.
    """
    engines = {}
    
    # 1. Calculate Factor Arrays for each Set
    for name, df in parts_data.items():
        # Only use common items for valid comparison
        df_common = df[common_cols]
        engine = QEngine(df_common, n_factors=n_factors, corr_method=corr_method).fit()
        engines[name] = engine.factor_arrays # (Items x Factors)
        
    results = []
    
    # 2. Compare Pairwise
    pairs = [('A','B'), ('A','C'), ('B','C')]
    for s1, s2 in pairs:
        if s1 not in engines or s2 not in engines: continue
        
        arr1 = engines[s1]
        arr2 = engines[s2]
        
        # Compare Factor 1 with Factor 1, 2 with 2...
        phis = []
        for f in range(n_factors):
            phi = tuckers_phi(arr1[:, f], arr2[:, f])
            phis.append(abs(phi)) # Absolute value for sign flip
            
        results.append({
            "Pair": f"{s1}-{s2}",
            "Mean Phi": np.mean(phis),
            "Factors": phis
        })
        
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
            # Identify ID col
            id_col = next((c for c in df.columns if str(c).lower() in ['email', 'id', 'respondent']), None)
            if not id_col:
                df['ID'] = [f"P{i}" for i in range(len(df))]
                id_col = 'ID'
            
            # Filter Q-sort columns (numeric)
            numeric_df = df.apply(pd.to_numeric, errors='coerce')
            # Keep columns with >50% valid numeric data
            valid_cols = numeric_df.columns[numeric_df.notna().sum() > len(df)*0.5]
            
            # Exclude ID col from data
            valid_cols = [c for c in valid_cols if c != id_col]
            
            final_df = numeric_df[valid_cols].copy()
            final_df.index = df[id_col]
            
            # Drop rows with too many NaNs
            final_df = final_df.dropna(thresh=len(valid_cols)*0.8)
            
            parts[sname.replace("PART", "")] = final_df
            
    return parts

def get_common_columns(parts):
    # Find columns starting with C/c followed by digits
    pat = re.compile(r"^C\d+$", re.IGNORECASE)
    sets_cols = []
    for df in parts.values():
        cols = {c for c in df.columns if pat.match(str(c))}
        sets_cols.append(cols)
    
    if not sets_cols: return []
    common = set.intersection(*sets_cols)
    return sorted(list(common))

# ==========================================
# 5. Main Execution
# ==========================================

st.title("Q-Methodology Refactored Analysis")
st.markdown("""
> **교수님을 위한 참고사항 (리커트 최적화 & 기능 추가):**
> - **Distinguishing Statements:** 각 요인을 다른 요인들과 구분 짓는 핵심 문항을 찾아주는 탭이 추가되었습니다.
> - **Noise Injection:** 부트스트랩 시 미세한 노이즈를 섞어(Robustness Test) 가짜 1.0 안정도를 방지합니다.
""")

uploaded_file = st.sidebar.file_uploader("Upload Excel (PARTA/B/C)", type='xlsx')

if uploaded_file:
    parts = parse_uploaded_file(uploaded_file)
    st.sidebar.success(f"Loaded Sets: {list(parts.keys())}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["1. Basic Q-Analysis", "2. Cross-Set Congruence", "3. Bootstrap Stability", "4. Distinguishing Statements"])
    
    # --- Tab 1: Basic Analysis ---
    with tab1:
        st.header("Single Set Analysis")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            target_set = st.selectbox("Select Set", list(parts.keys()))
        with c2:
            n_factors = st.number_input("Number of Factors", 1, 7, 3)
        with c3:
            corr_method = st.selectbox("Correlation Method", ["pearson", "spearman"], index=1, help="Spearman is recommended for Likert scales.")
        
        if target_set:
            df = parts[target_set]
            engine = QEngine(df, n_factors=n_factors, corr_method=corr_method).fit()
            
            st.info(f"Top 5 Eigenvalues: {np.round(engine.eigenvalues[:5], 2)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Explained Variance")
                st.write(engine.explained_variance)
            
            with col2:
                st.subheader("Significant Type Assignment (>0.4)")
                # Improved Assignment: Threshold based
                loadings = engine.loadings
                # Only count if max loading > 0.4
                max_vals = np.max(np.abs(loadings), axis=1)
                max_idxs = np.argmax(np.abs(loadings), axis=1)
                
                # Filter meaningless loadings
                valid_types = [f"Type {i+1}" if v > 0.4 else "None" for i, v in zip(max_idxs, max_vals)]
                counts = pd.Series(valid_types).value_counts().sort_index()
                st.write(counts)
                
            st.subheader("Factor Loadings (Person x Factor)")
            st.dataframe(pd.DataFrame(engine.loadings, index=df.index, columns=[f"F{i+1}" for i in range(n_factors)]).style.background_gradient(cmap="Blues"))
            
            st.subheader("Factor Arrays (Item Z-scores)")
            st.dataframe(pd.DataFrame(engine.factor_arrays, index=df.columns, columns=[f"F{i+1}" for i in range(n_factors)]).style.background_gradient(cmap="RdBu_r"))

    # --- Tab 2: Cross-Set Congruence ---
    with tab2:
        st.header("Cross-Set Congruence (Tucker's Phi)")
        common_cols = get_common_columns(parts)
        
        if len(common_cols) < 5:
            st.warning(f"Not enough common 'C' columns found. Found: {common_cols}")
        else:
            st.info(f"Analyzing using {len(common_cols)} common items: {common_cols[0]} ... {common_cols[-1]}")
            c1, c2 = st.columns(2)
            with c1:
                n_factors_cross = st.slider("Factors to Compare", 2, 5, 3, key='cross_k')
            with c2:
                corr_method_cross = st.selectbox("Correlation Method (Cross)", ["pearson", "spearman"], index=1, key='cross_c')

            results = calculate_cross_set_congruence(parts, common_cols, n_factors_cross, corr_method=corr_method_cross)
            
            for res in results:
                st.subheader(f"Congruence: {res['Pair']}")
                cols = st.columns(len(res['Factors']))
                for i, phi in enumerate(res['Factors']):
                    cols[i].metric(f"Factor {i+1}", f"{phi:.3f}")
                st.caption("Values > 0.90 indicate equivalence. > 0.80 indicate high similarity.")
                st.divider()

    # --- Tab 3: Bootstrap Stability ---
    with tab3:
        st.header("Bootstrap Stability Test")
        
        c1, c2 = st.columns(2)
        with c1:
            target_bs = st.selectbox("Set for Bootstrap", list(parts.keys()), key='bs_set')
            n_boot = st.number_input("Bootstrap Iterations", 50, 1000, 100)
        with c2:
            corr_method_bs = st.selectbox("Correlation Method (Boot)", ["pearson", "spearman"], index=1, key='bs_c')
            noise_level = st.slider("Noise Injection (Std Dev)", 0.0, 0.5, 0.05, 0.01, help="Add random noise to test structural robustness. 0.05-0.1 is recommended for Likert scales.")
        
        if st.button("Run Bootstrap"):
            with st.spinner("Resampling people and comparing factor arrays..."):
                # Use all items for internal stability
                res = bootstrap_stability(
                    parts[target_bs].values, 
                    n_factors=3, 
                    n_boot=n_boot, 
                    corr_method=corr_method_bs,
                    noise_std=noise_level
                )
            
            st.success("Analysis Complete")
            
            res_df = pd.DataFrame({
                "Mean Phi": res['mean'],
                "Std Dev": res['std'],
                "Stability Rate (>0.80)": res['rate_80'],
                "Stability Rate (>0.90)": res['rate_90']
            }, index=[f"Factor {i+1}" for i in range(len(res['mean']))])
            
            st.dataframe(res_df.style.format("{:.3f}").background_gradient(cmap="Greens", subset=["Stability Rate (>0.80)"]))
            st.markdown("""
            **해석 가이드:**
            * **Noise Injection:** 노이즈를 주입했음에도 Stability Rate가 높다면, 해당 요인은 매우 견고한 구조를 가진 것입니다.
            * **Mean Phi:** 노이즈가 있을 때 1.0보다 약간 낮게 나오는 것이 정상입니다 (0.90~0.98 등).
            """)

    # --- Tab 4: Distinguishing Statements ---
    with tab4:
        st.header("Distinguishing Statements Identification")
        st.caption("Identify items that statistically distinguish a factor from ALL other factors.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            target_set_dist = st.selectbox("Select Set", list(parts.keys()), key='dist_set')
        with c2:
            n_factors_dist = st.number_input("Number of Factors", 2, 7, 3, key='dist_nf')
        with c3:
            z_threshold = st.slider("Z-Score Difference Threshold", 0.5, 2.0, 1.0, 0.1, help="Difference in Z-score required to be considered distinguishing. Default 1.0 (approx 1 SD).")

        if target_set_dist:
            df = parts[target_set_dist]
            # Run engine to get arrays
            engine = QEngine(df, n_factors=n_factors_dist, corr_method=corr_method).fit() # Use method from Tab 1 or default? Use default Spearman from tab 1 if possible or just Spearman
            # For simplicity, using same corr_method from Tab 1 if available, else Spearman default in engine logic? No, let's just create new engine.
            # Ideally user selects correlation here too, but let's default to Spearman for Likert robustness.
            
            dist_dict = find_distinguishing_items(engine.factor_arrays, n_factors_dist, threshold=z_threshold, item_labels=df.columns)
            
            st.subheader(f"Results for {target_set_dist}")
            
            # Display per factor
            factor_tabs = st.tabs([f"Factor {i+1}" for i in range(n_factors_dist)])
            
            for i, tab in enumerate(factor_tabs):
                f_key = f"F{i+1}"
                with tab:
                    items_df = dist_dict.get(f_key)
                    if items_df is not None and not items_df.empty:
                        st.write(f"**{len(items_df)} distinguishing items found** (Threshold > {z_threshold})")
                        st.dataframe(items_df.style.background_gradient(cmap="coolwarm", subset=[f_key, "Difference"], vmin=-2, vmax=2))
                        st.markdown("""
                        * **Higher:** 이 요인이 다른 모든 요인보다 유의미하게 **더 높게** 평가한 항목
                        * **Lower:** 이 요인이 다른 모든 요인보다 유의미하게 **더 낮게** 평가한 항목
                        """)
                    else:
                        st.info("No distinguishing items found at this threshold. Try lowering the Z-Score difference.")

else:
    st.info("Please upload the Excel file to begin.")
