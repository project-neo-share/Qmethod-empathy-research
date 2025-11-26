# -*- coding: utf-8 -*-
"""
Q-Methodology Analysis Engine (Refactored for Accuracy)
- Author: Prof. Dr. SongheeKang
- Reference: Brown, S. R. (1980). Political subjectivity.
- Key Fix: Comparing 'Factor Arrays' (Item Z-scores) for stability/congruence, not Person Loadings.
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
    def __init__(self, data_df, n_factors=3, rotation=True):
        self.raw_df = data_df
        # Ensure numeric
        self.data = data_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.n_persons, self.n_items = self.data.shape
        self.n_factors = n_factors
        self.rotation = rotation
        self.loadings = None
        self.factor_arrays = None # (n_items x n_factors)
        self.explained_variance = None
        
    def fit(self):
        # 1. Correlation Matrix (Person x Person)
        # Using Pearson on standardized sorts
        z_data = standardize_rows(self.data)
        R = np.corrcoef(z_data)
        
        # 2. Eigen Decomposition (Centroid/PCA approx)
        eigvals, eigvecs = np.linalg.eigh(R)
        # Sort descending
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # 3. Extract Factors
        # Loadings = Eigenvector * sqrt(Eigenvalue)
        k = self.n_factors
        L = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        
        # 4. Varimax Rotation
        if self.rotation and k > 1:
            L = self._varimax(L)
            
        self.loadings = L
        self.explained_variance = eigvals[:k]
        
        # 5. Calculate Factor Arrays (The Critical Step)
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
        """
        n_items = z_data.shape[1]
        arrays = np.zeros((n_items, self.n_factors))
        
        for f in range(self.n_factors):
            # Only use significant loaders for clean arrays? 
            # Classic method uses all weighted, or flag-based. 
            # We will use weighted average of all to ensure stability in bootstrap.
            
            l_vec = loadings[:, f]
            
            # Weight calculation (Brown, 1980)
            # Handle 1.0 or -1.0 loadings to avoid infinity
            l_clean = np.clip(l_vec, -0.999, 0.999)
            weights = l_clean / (1 - l_clean**2)
            
            # Weighted average of item z-scores
            # (Weights * Person_Sorts) / Sum_Weights ? 
            # Formula: Z_j = sum(w_i * z_ij) / sqrt(sum(w_i^2))
            
            w_abs_sum = np.sum(np.abs(weights))
            if w_abs_sum == 0:
                arrays[:, f] = 0
                continue
                
            # Weighted sum of sorts
            # (n_persons) dot (n_persons, n_items) -> (n_items)
            weighted_sum = np.dot(weights, z_data)
            
            # Normalize to Z-scores
            # Standard Error of factor scores = sqrt(sum(w^2)) is one way,
            # but standardizing the final array to (0,1) is safer for comparison.
            
            arr_mean = np.mean(weighted_sum)
            arr_std = np.std(weighted_sum, ddof=1)
            if arr_std == 0: arr_std = 1.0
            
            arrays[:, f] = (weighted_sum - arr_mean) / arr_std
            
        return arrays # (Items x Factors)

# ==========================================
# 3. Stability & Congruence Logic
# ==========================================

@st.cache_data(show_spinner=False)
def bootstrap_stability(df_values, n_factors=3, n_boot=200):
    """
    Checks if the Factor Arrays (Item profiles) remain consistent
    when people are resampled.
    """
    # 1. Original Solution
    base_engine = QEngine(pd.DataFrame(df_values), n_factors=n_factors).fit()
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
        
        # Run Q-Analysis on Sample
        boot_engine = QEngine(sample_data, n_factors=n_factors).fit()
        boot_arrays = boot_engine.factor_arrays
        
        # Compare Factors (Greedy Match or Procrustes)
        # We assume F1 matches F1 generally, but sign might flip.
        # Or order might swap. To be rigorous, we match to closest.
        
        for f in range(n_factors):
            target = base_arrays[:, f]
            
            # Find best match in boot_arrays
            best_phi = -1.0
            for bf in range(n_factors):
                candidate = boot_arrays[:, bf]
                phi = tuckers_phi(target, candidate)
                # Handle sign indeterminacy (Factor could be inverted)
                if abs(phi) > abs(best_phi):
                    best_phi = phi
            
            # We take the absolute congruence because sign flip is trivial in Q
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
def calculate_cross_set_congruence(parts_data, common_cols, n_factors=3):
    """
    Calculates Factor Arrays for Set A, B, C separately,
    then calculates Phi between them.
    """
    engines = {}
    
    # 1. Calculate Factor Arrays for each Set
    for name, df in parts_data.items():
        # Only use common items for valid comparison
        df_common = df[common_cols]
        engine = QEngine(df_common, n_factors=n_factors).fit()
        engines[name] = engine.factor_arrays # (Items x Factors)
        
    results = []
    
    # 2. Compare Pairwise
    pairs = [('A','B'), ('A','C'), ('B','C')]
    for s1, s2 in pairs:
        if s1 not in engines or s2 not in engines: continue
        
        arr1 = engines[s1]
        arr2 = engines[s2]
        
        # Compare Factor 1 with Factor 1, 2 with 2...
        # (Assuming factors order similarly, which they usually do if dominant)
        phis = []
        for f in range(n_factors):
            # Check F_f in Set1 vs F_f in Set2
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
            # Regex for Q-sort items (assuming 'Q1', 'C1', '1', etc.) or just numeric
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
> **교수님을 위한 참고사항:**
> 이 버전은 Q방법론의 **Factor Array(문항 배열)**를 직접 산출하여 비교하도록 완전히 재설계되었습니다.
> - **안정도(Bootstrap):** 사람이 바뀌어도 '문항의 배열'이 유지되는지 확인합니다.
> - **일치도(Congruence):** Set A와 Set B의 '문항 배열'이 얼마나 유사한지(Phi) 계산합니다.
> - **수치 해석:** 0.80 이상이면 신뢰할 수 있는 수준입니다.
""")

uploaded_file = st.sidebar.file_uploader("Upload Excel (PARTA/B/C)", type='xlsx')

if uploaded_file:
    parts = parse_uploaded_file(uploaded_file)
    st.sidebar.success(f"Loaded Sets: {list(parts.keys())}")
    
    tab1, tab2, tab3 = st.tabs(["1. Basic Q-Analysis", "2. Cross-Set Congruence", "3. Bootstrap Stability"])
    
    # --- Tab 1: Basic Analysis ---
    with tab1:
        st.header("Single Set Analysis")
        target_set = st.selectbox("Select Set", list(parts.keys()))
        n_factors = st.number_input("Number of Factors", 1, 7, 3)
        
        if target_set:
            df = parts[target_set]
            engine = QEngine(df, n_factors=n_factors).fit()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Explained Variance")
                st.write(engine.explained_variance)
            
            with col2:
                st.subheader("Type Assignment")
                # Simple Max Loading Assignment
                loadings = engine.loadings
                max_idx = np.argmax(np.abs(loadings), axis=1)
                max_val = np.max(np.abs(loadings), axis=1)
                counts = pd.Series(max_idx).value_counts().sort_index()
                counts.index = [f"Type {i+1}" for i in counts.index]
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
            n_factors_cross = st.slider("Factors to Compare", 2, 5, 3, key='cross_k')
            
            results = calculate_cross_set_congruence(parts, common_cols, n_factors_cross)
            
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
        target_bs = st.selectbox("Set for Bootstrap", list(parts.keys()), key='bs_set')
        n_boot = st.number_input("Bootstrap Iterations", 50, 1000, 100)
        
        if st.button("Run Bootstrap"):
            with st.spinner("Resampling people and comparing factor arrays..."):
                # Use all items for internal stability
                res = bootstrap_stability(parts[target_bs].values, n_factors=3, n_boot=n_boot)
            
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
            * **Mean Phi:** 재추출된 요인과 원본 요인 간의 평균 유사도입니다. (1.0에 가까울수록 좋음)
            * **Stability Rate:** 부트스트랩 반복 중 80% 이상의 경우에서 요인이 유지된 비율입니다. 이 값이 1.0(100%)에 가까워야 '견고한 요인'입니다.
            """)

else:
    st.info("Please upload the Excel file to begin.")
