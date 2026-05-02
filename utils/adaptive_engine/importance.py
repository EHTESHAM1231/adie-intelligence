"""
ADIE — Feature Importance Analyzer
===================================

Pre-cleaning feature importance analysis to identify and protect
critical features before any transformations are applied.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats as scipy_stats
import warnings

warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance BEFORE cleaning to identify columns
    that must be protected from aggressive transformations.
    
    Protection levels:
    - PROTECTED: High importance - cannot be degraded
    - CAREFUL: Medium importance - transform with care
    - FLEXIBLE: Low importance - can be transformed freely
    """
    
    # Importance thresholds
    THRESHOLDS = {
        "protected": 0.7,   # Top 30% of features
        "careful": 0.4,     # Middle 30%
        "flexible": 0.0,    # Bottom 40%
    }
    
    def analyze_importance(
        self,
        df: pd.DataFrame,
        target_col: str,
        profile: Optional[Dict] = None,
        max_features: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze feature importance for all columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        profile : dict, optional
            Dataset profile from profiler
        max_features : int
            Maximum features to analyze (for performance)
            
        Returns
        -------
        {
            "importance_scores": {col: score},
            "protection_levels": {col: level},
            "protected_columns": [cols],
            "careful_columns": [cols],
            "flexible_columns": [cols],
            "analysis_method": str,
            "task_type": str
        }
        """
        df = df.copy()
        df.columns = df.columns.str.strip()
        target_col = target_col.strip()
        
        if target_col not in df.columns:
            return self._empty_result(f"Target column '{target_col}' not found")
        
        # Determine task type
        task_type = self._determine_task_type(df[target_col])
        
        # Prepare data for analysis
        X, y, feature_names, prep_info = self._prepare_data(
            df, target_col, max_features
        )
        
        if X is None or len(feature_names) == 0:
            return self._empty_result("Could not prepare data for importance analysis")
        
        # Calculate importance scores
        importance_scores, method = self._calculate_importance(X, y, task_type)
        
        # Map scores back to original column names
        col_scores = {}
        for i, col in enumerate(feature_names):
            if i < len(importance_scores):
                col_scores[col] = float(importance_scores[i])
        
        # Normalize scores to [0, 1]
        if col_scores:
            max_score = max(col_scores.values())
            min_score = min(col_scores.values())
            score_range = max_score - min_score
            if score_range > 0:
                col_scores = {
                    col: (score - min_score) / score_range
                    for col, score in col_scores.items()
                }
        
        # Assign protection levels
        protection_levels = self._assign_protection_levels(col_scores)
        
        # Group columns by protection level
        protected = [col for col, level in protection_levels.items() if level == "PROTECTED"]
        careful = [col for col, level in protection_levels.items() if level == "CAREFUL"]
        flexible = [col for col, level in protection_levels.items() if level == "FLEXIBLE"]
        
        return {
            "importance_scores": {k: round(v, 4) for k, v in col_scores.items()},
            "protection_levels": protection_levels,
            "protected_columns": protected,
            "careful_columns": careful,
            "flexible_columns": flexible,
            "analysis_method": method,
            "task_type": task_type,
            "n_features_analyzed": len(feature_names),
            "preparation_info": prep_info,
        }
    
    def _determine_task_type(self, target: pd.Series) -> str:
        """Determine if this is classification or regression."""
        n_unique = target.nunique()
        
        if pd.api.types.is_numeric_dtype(target) and n_unique > 20:
            return "regression"
        else:
            return "classification"
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        max_features: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], Dict]:
        """
        Prepare data for importance analysis.
        Handles missing values and encoding without dropping columns.
        """
        prep_info = {
            "columns_encoded": [],
            "columns_imputed": [],
            "columns_excluded": [],
        }
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col]
        
        # Limit features for performance
        if len(feature_cols) > max_features:
            # Prioritize numeric columns
            numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
            categorical_cols = [c for c in feature_cols if c not in numeric_cols]
            
            # Take all numeric + some categorical
            n_cat = max_features - len(numeric_cols)
            if n_cat > 0:
                feature_cols = numeric_cols + categorical_cols[:n_cat]
            else:
                feature_cols = numeric_cols[:max_features]
        
        if len(feature_cols) == 0:
            return None, None, [], prep_info
        
        # Prepare feature matrix
        X_parts = []
        final_feature_names = []
        
        for col in feature_cols:
            series = df[col].copy()
            
            # Handle missing values
            if series.isnull().any():
                prep_info["columns_imputed"].append(col)
            
            if pd.api.types.is_numeric_dtype(series):
                # Numeric: fill with median
                numeric_series = pd.to_numeric(series, errors='coerce')
                median_val = numeric_series.median()
                if pd.isna(median_val):
                    median_val = 0
                numeric_series = numeric_series.fillna(median_val)
                X_parts.append(numeric_series.values.reshape(-1, 1))
                final_feature_names.append(col)
            
            elif series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
                # Categorical: label encode
                try:
                    le = LabelEncoder()
                    # Fill missing with placeholder
                    series = series.fillna("__MISSING__")
                    encoded = le.fit_transform(series.astype(str))
                    X_parts.append(encoded.reshape(-1, 1))
                    final_feature_names.append(col)
                    prep_info["columns_encoded"].append(col)
                except Exception:
                    prep_info["columns_excluded"].append(col)
            
            else:
                prep_info["columns_excluded"].append(col)
        
        if len(X_parts) == 0:
            return None, None, [], prep_info
        
        X = np.hstack(X_parts)
        
        # Prepare target
        y = df[target_col].copy()
        
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            le = LabelEncoder()
            y = y.fillna("__MISSING__")
            y = le.fit_transform(y.astype(str))
        else:
            y_numeric = pd.to_numeric(y, errors='coerce')
            median_val = y_numeric.median()
            y = y_numeric.fillna(median_val if not pd.isna(median_val) else 0)
            y = y.values
        
        # Remove rows where target is still problematic
        valid_mask = ~np.isnan(y) if np.issubdtype(y.dtype, np.floating) else np.ones(len(y), dtype=bool)
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y, final_feature_names, prep_info
    
    def _calculate_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str
    ) -> Tuple[np.ndarray, str]:
        """
        Calculate feature importance using appropriate method.
        """
        n_samples, n_features = X.shape
        
        # Try mutual information first (works for both tasks)
        try:
            if task_type == "classification":
                # Ensure y is integer for classification
                y_int = y.astype(int)
                scores = mutual_info_classif(
                    X, y_int,
                    discrete_features='auto',
                    random_state=42,
                    n_neighbors=min(5, n_samples - 1)
                )
                return scores, "mutual_information_classification"
            else:
                scores = mutual_info_regression(
                    X, y,
                    discrete_features='auto',
                    random_state=42,
                    n_neighbors=min(5, n_samples - 1)
                )
                return scores, "mutual_information_regression"
        except Exception as e:
            pass
        
        # Fallback to correlation-based importance
        try:
            if task_type == "regression":
                scores = np.array([
                    abs(np.corrcoef(X[:, i], y)[0, 1])
                    for i in range(n_features)
                ])
                scores = np.nan_to_num(scores, 0)
                return scores, "correlation"
            else:
                # For classification, use ANOVA F-statistic
                scores = []
                for i in range(n_features):
                    try:
                        f_stat, _ = scipy_stats.f_oneway(
                            *[X[y == c, i] for c in np.unique(y) if len(X[y == c, i]) > 0]
                        )
                        scores.append(f_stat if not np.isnan(f_stat) else 0)
                    except:
                        scores.append(0)
                return np.array(scores), "anova_f_statistic"
        except Exception:
            pass
        
        # Last resort: variance-based importance
        try:
            scores = np.var(X, axis=0)
            scores = scores / (scores.max() + 1e-10)
            return scores, "variance"
        except:
            return np.ones(n_features) / n_features, "uniform"
    
    def _assign_protection_levels(self, col_scores: Dict[str, float]) -> Dict[str, str]:
        """Assign protection levels based on importance scores."""
        if not col_scores:
            return {}
        
        # Sort columns by score
        sorted_cols = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
        n_cols = len(sorted_cols)
        
        # Calculate thresholds based on percentiles
        scores = [s for _, s in sorted_cols]
        p70 = np.percentile(scores, 70) if len(scores) > 0 else 0.7
        p40 = np.percentile(scores, 40) if len(scores) > 0 else 0.4
        
        protection_levels = {}
        for col, score in sorted_cols:
            if score >= p70:
                protection_levels[col] = "PROTECTED"
            elif score >= p40:
                protection_levels[col] = "CAREFUL"
            else:
                protection_levels[col] = "FLEXIBLE"
        
        return protection_levels
    
    def _empty_result(self, error_msg: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        return {
            "importance_scores": {},
            "protection_levels": {},
            "protected_columns": [],
            "careful_columns": [],
            "flexible_columns": [],
            "analysis_method": "none",
            "task_type": "unknown",
            "error": error_msg,
        }
    
    def get_protection_summary(self, importance_result: Dict) -> str:
        """Generate a human-readable protection summary."""
        protected = importance_result.get("protected_columns", [])
        careful = importance_result.get("careful_columns", [])
        flexible = importance_result.get("flexible_columns", [])
        
        summary = []
        summary.append(f"Feature Protection Analysis ({importance_result.get('analysis_method', 'unknown')})")
        summary.append("=" * 50)
        summary.append(f"Task Type: {importance_result.get('task_type', 'unknown')}")
        summary.append(f"Features Analyzed: {importance_result.get('n_features_analyzed', 0)}")
        summary.append("")
        summary.append(f"🛡️ PROTECTED ({len(protected)} columns):")
        for col in protected[:10]:
            score = importance_result["importance_scores"].get(col, 0)
            summary.append(f"   - {col}: {score:.4f}")
        if len(protected) > 10:
            summary.append(f"   ... and {len(protected) - 10} more")
        
        summary.append(f"\n⚠️ CAREFUL ({len(careful)} columns):")
        for col in careful[:5]:
            score = importance_result["importance_scores"].get(col, 0)
            summary.append(f"   - {col}: {score:.4f}")
        if len(careful) > 5:
            summary.append(f"   ... and {len(careful) - 5} more")
        
        summary.append(f"\n✅ FLEXIBLE ({len(flexible)} columns):")
        for col in flexible[:5]:
            score = importance_result["importance_scores"].get(col, 0)
            summary.append(f"   - {col}: {score:.4f}")
        if len(flexible) > 5:
            summary.append(f"   ... and {len(flexible) - 5} more")
        
        return "\n".join(summary)
