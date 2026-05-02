"""
ADIE — Dataset Profiling Engine
================================

Comprehensive profiling system that extracts all characteristics needed
for intelligent, adaptive data preparation decisions.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List, Any, Optional
import sys


class DatasetProfiler:
    """
    Comprehensive dataset profiler that extracts all characteristics
    needed for downstream adaptive decisions.
    """
    
    def __init__(self):
        self.profile = {}
    
    def profile_dataset(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_col : str, optional
            Target column name (auto-detect if None)
            
        Returns
        -------
        dict
            Comprehensive dataset profile
        """
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        if target_col:
            target_col = target_col.strip()
        
        n_rows, n_cols = df.shape
        
        profile = {
            # Basic dimensions
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cells": n_rows * n_cols,
            
            # Size classification
            "size_class": self._classify_size(n_rows),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            
            # Column lists by type
            "columns": df.columns.tolist(),
            "numeric_cols": df.select_dtypes(include=np.number).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "boolean_cols": df.select_dtypes(include='bool').columns.tolist(),
            "datetime_cols": self._detect_datetime_columns(df),
            
            # Per-column analysis
            "column_profiles": self._profile_all_columns(df, target_col),
            
            # Missing value analysis
            "missing_analysis": self._analyze_missing(df),
            
            # Cardinality analysis
            "cardinality_analysis": self._analyze_cardinality(df),
            
            # Distribution analysis
            "distribution_analysis": self._analyze_distributions(df),
            
            # Correlation analysis
            "correlation_analysis": self._analyze_correlations(df),
            
            # Duplicate analysis
            "duplicate_analysis": self._analyze_duplicates(df),
            
            # Target variable analysis
            "target_analysis": self._analyze_target(df, target_col) if target_col else None,
            
            # Data quality score
            "quality_metrics": self._compute_quality_metrics(df),
            
            # Structural patterns
            "structural_patterns": self._detect_structural_patterns(df),
        }
        
        self.profile = profile
        return profile
    
    def _classify_size(self, n_rows: int) -> str:
        """Classify dataset size."""
        if n_rows < 1000:
            return "small"
        elif n_rows < 100000:
            return "medium"
        else:
            return "large"
    
    def _detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect datetime columns including string-formatted dates."""
        datetime_cols = []
        
        for col in df.columns:
            # Already datetime type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
                continue
            
            # Check string columns for date patterns
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(20)
                if len(sample) == 0:
                    continue
                
                # Try parsing as datetime
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    valid_ratio = parsed.notna().sum() / len(sample)
                    if valid_ratio > 0.8:
                        datetime_cols.append(col)
                except:
                    pass
        
        return datetime_cols
    
    def _profile_all_columns(self, df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Dict]:
        """Generate detailed profile for each column."""
        profiles = {}
        
        for col in df.columns:
            profiles[col] = self._profile_single_column(df[col], col, col == target_col)
        
        return profiles
    
    def _profile_single_column(self, series: pd.Series, col_name: str, is_target: bool) -> Dict:
        """Profile a single column comprehensively."""
        n_total = len(series)
        n_missing = int(series.isnull().sum())
        n_unique = int(series.nunique())
        
        profile = {
            "name": col_name,
            "dtype": str(series.dtype),
            "is_target": is_target,
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_ratio": round(n_missing / n_total, 4) if n_total > 0 else 0,
            "n_unique": n_unique,
            "unique_ratio": round(n_unique / n_total, 4) if n_total > 0 else 0,
            "n_duplicates": n_total - n_unique,
        }
        
        # Numeric-specific stats
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                profile.update({
                    "mean": round(float(non_null.mean()), 4),
                    "median": round(float(non_null.median()), 4),
                    "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0,
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "q1": float(non_null.quantile(0.25)),
                    "q3": float(non_null.quantile(0.75)),
                    "skewness": round(float(non_null.skew()), 4) if len(non_null) > 2 else 0,
                    "kurtosis": round(float(non_null.kurtosis()), 4) if len(non_null) > 3 else 0,
                    "has_negative": bool((non_null < 0).any()),
                    "has_zero": bool((non_null == 0).any()),
                    "is_integer_like": bool(np.allclose(non_null, non_null.round())),
                })
                
                # Outlier detection
                q1, q3 = non_null.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                n_outliers = int(((non_null < lower_bound) | (non_null > upper_bound)).sum())
                profile["n_outliers"] = n_outliers
                profile["outlier_ratio"] = round(n_outliers / len(non_null), 4)
        
        # Categorical-specific stats
        elif series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
            value_counts = series.value_counts()
            profile.update({
                "top_values": value_counts.head(10).to_dict(),
                "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "avg_string_length": round(series.dropna().astype(str).str.len().mean(), 2),
                "max_string_length": int(series.dropna().astype(str).str.len().max()) if len(series.dropna()) > 0 else 0,
            })
            
            # Cardinality classification
            if n_unique <= 2:
                profile["cardinality_class"] = "binary"
            elif n_unique <= 10:
                profile["cardinality_class"] = "low"
            elif n_unique <= 50:
                profile["cardinality_class"] = "medium"
            elif n_unique <= 500:
                profile["cardinality_class"] = "high"
            else:
                profile["cardinality_class"] = "very_high"
        
        return profile
    
    def _analyze_missing(self, df: pd.DataFrame) -> Dict:
        """Comprehensive missing value analysis."""
        missing_per_col = df.isnull().sum()
        missing_per_row = df.isnull().sum(axis=1)
        
        total_missing = int(missing_per_col.sum())
        total_cells = df.shape[0] * df.shape[1]
        
        # Classify columns by missing ratio
        cols_no_missing = []
        cols_low_missing = []      # < 5%
        cols_medium_missing = []   # 5-30%
        cols_high_missing = []     # 30-80%
        cols_very_high_missing = [] # > 80%
        
        for col in df.columns:
            ratio = missing_per_col[col] / len(df)
            if ratio == 0:
                cols_no_missing.append(col)
            elif ratio < 0.05:
                cols_low_missing.append(col)
            elif ratio < 0.30:
                cols_medium_missing.append(col)
            elif ratio < 0.80:
                cols_high_missing.append(col)
            else:
                cols_very_high_missing.append(col)
        
        return {
            "total_missing": total_missing,
            "total_cells": total_cells,
            "overall_missing_ratio": round(total_missing / total_cells, 4) if total_cells > 0 else 0,
            "missing_per_column": {k: int(v) for k, v in missing_per_col.to_dict().items()},
            "missing_ratio_per_column": {k: round(v / len(df), 4) for k, v in missing_per_col.to_dict().items()},
            "rows_with_any_missing": int((missing_per_row > 0).sum()),
            "rows_complete": int((missing_per_row == 0).sum()),
            "cols_no_missing": cols_no_missing,
            "cols_low_missing": cols_low_missing,
            "cols_medium_missing": cols_medium_missing,
            "cols_high_missing": cols_high_missing,
            "cols_very_high_missing": cols_very_high_missing,
            "missing_pattern": self._detect_missing_pattern(df),
        }
    
    def _detect_missing_pattern(self, df: pd.DataFrame) -> str:
        """Detect the pattern of missing values."""
        missing_matrix = df.isnull()
        
        # Check if missing values are random or structured
        if missing_matrix.sum().sum() == 0:
            return "none"
        
        # Check for column-wise pattern (entire columns mostly missing)
        col_missing_ratios = missing_matrix.mean()
        if (col_missing_ratios > 0.8).any():
            return "column_concentrated"
        
        # Check for row-wise pattern
        row_missing_ratios = missing_matrix.mean(axis=1)
        if (row_missing_ratios > 0.5).sum() > len(df) * 0.1:
            return "row_concentrated"
        
        # Check for correlation between missing values
        if missing_matrix.shape[1] > 1:
            missing_corr = missing_matrix.corr()
            high_corr = (missing_corr.abs() > 0.7).sum().sum() - len(missing_corr)
            if high_corr > 0:
                return "correlated"
        
        return "random"
    
    def _analyze_cardinality(self, df: pd.DataFrame) -> Dict:
        """Analyze cardinality patterns across the dataset."""
        cardinality = {}
        
        for col in df.columns:
            n_unique = df[col].nunique()
            ratio = n_unique / len(df) if len(df) > 0 else 0
            
            cardinality[col] = {
                "n_unique": int(n_unique),
                "unique_ratio": round(ratio, 4),
                "is_constant": n_unique <= 1,
                "is_binary": n_unique == 2,
                "is_identifier_like": ratio > 0.95 and n_unique > 100,
            }
        
        # Identify potential identifiers
        potential_identifiers = [
            col for col, stats in cardinality.items()
            if stats["is_identifier_like"]
        ]
        
        # Identify constant columns
        constant_columns = [
            col for col, stats in cardinality.items()
            if stats["is_constant"]
        ]
        
        return {
            "per_column": cardinality,
            "potential_identifiers": potential_identifiers,
            "constant_columns": constant_columns,
            "binary_columns": [col for col, stats in cardinality.items() if stats["is_binary"]],
            "high_cardinality_columns": [
                col for col, stats in cardinality.items()
                if stats["n_unique"] > 50 and not stats["is_identifier_like"]
            ],
        }
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict:
        """Analyze distributions of numeric columns."""
        distributions = {}
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            
            skewness = float(series.skew())
            kurtosis = float(series.kurtosis())
            
            # Classify distribution shape
            if abs(skewness) < 0.5:
                skew_class = "symmetric"
            elif skewness > 0:
                skew_class = "right_skewed" if skewness < 2 else "heavily_right_skewed"
            else:
                skew_class = "left_skewed" if skewness > -2 else "heavily_left_skewed"
            
            # Test for normality (only for reasonable sample sizes)
            is_normal = False
            if 20 <= len(series) <= 5000:
                try:
                    _, p_value = scipy_stats.normaltest(series)
                    is_normal = p_value > 0.05
                except:
                    pass
            
            distributions[col] = {
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "skew_class": skew_class,
                "is_approximately_normal": is_normal,
                "range": float(series.max() - series.min()),
                "coefficient_of_variation": round(float(series.std() / series.mean()), 4) if series.mean() != 0 else None,
            }
        
        # Identify highly skewed columns
        highly_skewed = [
            col for col, stats in distributions.items()
            if abs(stats["skewness"]) > 2
        ]
        
        return {
            "per_column": distributions,
            "highly_skewed_columns": highly_skewed,
            "normal_columns": [col for col, stats in distributions.items() if stats["is_approximately_normal"]],
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between numeric features."""
        numeric_df = df.select_dtypes(include=np.number)
        
        if numeric_df.shape[1] < 2:
            return {
                "correlation_matrix": {},
                "high_correlations": [],
                "correlation_density": 0.0,
            }
        
        try:
            corr_matrix = numeric_df.corr()
            
            # Find high correlations
            high_correlations = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Upper triangle only
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_correlations.append({
                                "col1": col1,
                                "col2": col2,
                                "correlation": round(float(corr_val), 4),
                            })
            
            # Sort by absolute correlation
            high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Calculate correlation density
            upper_triangle = np.triu(corr_matrix.abs().values, k=1)
            n_pairs = (corr_matrix.shape[0] * (corr_matrix.shape[0] - 1)) / 2
            n_high = (upper_triangle > 0.7).sum()
            correlation_density = n_high / n_pairs if n_pairs > 0 else 0
            
            return {
                "correlation_matrix": corr_matrix.round(4).to_dict(),
                "high_correlations": high_correlations[:20],  # Top 20
                "correlation_density": round(float(correlation_density), 4),
                "n_high_correlation_pairs": len(high_correlations),
            }
        except Exception as e:
            return {
                "correlation_matrix": {},
                "high_correlations": [],
                "correlation_density": 0.0,
                "error": str(e),
            }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate rows."""
        n_duplicates = int(df.duplicated().sum())
        n_rows = len(df)
        
        return {
            "n_duplicate_rows": n_duplicates,
            "duplicate_ratio": round(n_duplicates / n_rows, 4) if n_rows > 0 else 0,
            "n_unique_rows": n_rows - n_duplicates,
            "has_significant_duplicates": n_duplicates / n_rows > 0.05 if n_rows > 0 else False,
        }
    
    def _analyze_target(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze target variable characteristics."""
        if target_col not in df.columns:
            return {"error": f"Target column '{target_col}' not found"}
        
        target = df[target_col]
        n_unique = int(target.nunique())
        n_missing = int(target.isnull().sum())
        
        analysis = {
            "column": target_col,
            "dtype": str(target.dtype),
            "n_unique": n_unique,
            "n_missing": n_missing,
            "missing_ratio": round(n_missing / len(target), 4) if len(target) > 0 else 0,
        }
        
        # Determine task type
        if pd.api.types.is_numeric_dtype(target) and n_unique > 20:
            analysis["task_type"] = "regression"
            non_null = target.dropna()
            if len(non_null) > 0:
                analysis.update({
                    "mean": round(float(non_null.mean()), 4),
                    "std": round(float(non_null.std()), 4),
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                })
        else:
            analysis["task_type"] = "classification"
            value_counts = target.value_counts()
            
            if len(value_counts) > 0:
                majority_class = value_counts.index[0]
                minority_class = value_counts.index[-1]
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1] if value_counts.iloc[-1] > 0 else float('inf')
                
                analysis.update({
                    "n_classes": n_unique,
                    "class_distribution": {str(k): int(v) for k, v in value_counts.to_dict().items()},
                    "majority_class": str(majority_class),
                    "minority_class": str(minority_class),
                    "imbalance_ratio": round(float(imbalance_ratio), 2),
                    "is_imbalanced": imbalance_ratio > 3,
                    "is_binary": n_unique == 2,
                })
        
        return analysis
    
    def _compute_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute overall data quality metrics."""
        n_rows, n_cols = df.shape
        n_cells = n_rows * n_cols
        
        # Completeness
        n_missing = df.isnull().sum().sum()
        completeness = 1 - (n_missing / n_cells) if n_cells > 0 else 1
        
        # Uniqueness (row-level)
        n_duplicates = df.duplicated().sum()
        uniqueness = 1 - (n_duplicates / n_rows) if n_rows > 0 else 1
        
        # Consistency (no mixed types in columns)
        consistency_issues = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed numeric/text
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
                    if 0 < numeric_count < len(sample):
                        consistency_issues += 1
        consistency = 1 - (consistency_issues / n_cols) if n_cols > 0 else 1
        
        # Overall quality score
        quality_score = round((completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3) * 100, 1)
        
        return {
            "completeness": round(completeness, 4),
            "uniqueness": round(uniqueness, 4),
            "consistency": round(consistency, 4),
            "overall_quality_score": quality_score,
            "n_consistency_issues": consistency_issues,
        }
    
    def _detect_structural_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect structural patterns in the dataset."""
        patterns = {
            "has_id_column": False,
            "has_timestamp": False,
            "has_geographic": False,
            "has_text_heavy": False,
            "column_naming_convention": "unknown",
        }
        
        col_names_lower = [c.lower() for c in df.columns]
        
        # ID column detection
        id_keywords = ['id', '_id', 'key', 'code', 'number', 'no']
        patterns["has_id_column"] = any(
            any(kw in col for kw in id_keywords)
            for col in col_names_lower
        )
        
        # Timestamp detection
        time_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month']
        patterns["has_timestamp"] = any(
            any(kw in col for kw in time_keywords)
            for col in col_names_lower
        )
        
        # Geographic detection
        geo_keywords = ['lat', 'lon', 'city', 'state', 'country', 'zip', 'postal', 'address', 'location']
        patterns["has_geographic"] = any(
            any(kw in col for kw in geo_keywords)
            for col in col_names_lower
        )
        
        # Text-heavy detection
        text_cols = df.select_dtypes(include='object').columns
        if len(text_cols) > 0:
            avg_lengths = df[text_cols].apply(lambda x: x.astype(str).str.len().mean())
            patterns["has_text_heavy"] = (avg_lengths > 100).any()
        
        # Naming convention detection
        if all('_' in c for c in df.columns if len(c) > 3):
            patterns["column_naming_convention"] = "snake_case"
        elif all(c[0].isupper() for c in df.columns if len(c) > 0):
            patterns["column_naming_convention"] = "PascalCase"
        elif all(c[0].islower() and any(ch.isupper() for ch in c) for c in df.columns if len(c) > 1):
            patterns["column_naming_convention"] = "camelCase"
        
        return patterns
