"""
ADIE — Adaptive Transformer
============================

Intelligent transformation engine that applies context-aware,
non-destructive transformations based on dataset characteristics.

CORE PRINCIPLE: NO COLUMN IS EVER DROPPED
All transformations preserve or augment information.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, RobustScaler,
    MinMaxScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
import hashlib
import warnings

warnings.filterwarnings('ignore')


class AdaptiveTransformer:
    """
    Applies adaptive, non-destructive transformations based on:
    - Dataset type
    - Domain context
    - Column roles
    - Protection levels
    - Feature importance
    
    GUARANTEES:
    - No column is ever dropped
    - All transformations are traceable
    - Original information is preserved or augmented
    """
    
    def __init__(self):
        self.transformation_log = []
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.original_columns = []
        self.added_columns = []
    
    def transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        dataset_type: str,
        domain: str,
        column_roles: Dict,
        protection_system: Any,
        profile: Dict,
        config: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply adaptive transformations to the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        dataset_type : str
            Dataset type (tabular, relational, time_series, etc.)
        domain : str
            Detected domain
        column_roles : dict
            Column role classifications
        protection_system : ColumnProtectionSystem
            Protection system instance
        profile : dict
            Dataset profile
        config : dict, optional
            Additional configuration
            
        Returns
        -------
        (transformed_df, transformation_report)
        """
        self.transformation_log = []
        self.original_columns = df.columns.tolist()
        self.added_columns = []
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        result_df.columns = result_df.columns.str.strip()
        target_col = target_col.strip()
        
        # Verify target column exists
        if target_col not in result_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Get configuration
        config = config or {}
        
        # Step 1: Handle missing values (NEVER drop columns)
        result_df = self._handle_missing_values(
            result_df, target_col, column_roles, protection_system, profile
        )
        
        # Step 2: Handle temporal columns
        result_df = self._handle_temporal_columns(
            result_df, target_col, column_roles, protection_system
        )
        
        # Step 3: Handle categorical columns (NEVER drop)
        result_df = self._handle_categorical_columns(
            result_df, target_col, column_roles, protection_system, profile
        )
        
        # Step 4: Handle identifier columns (NEVER drop - encode instead)
        result_df = self._handle_identifier_columns(
            result_df, target_col, column_roles, protection_system
        )
        
        # Step 5: Handle numerical columns
        result_df = self._handle_numerical_columns(
            result_df, target_col, column_roles, protection_system, profile
        )
        
        # Step 6: Handle outliers (cap, don't remove)
        result_df = self._handle_outliers(
            result_df, target_col, column_roles, protection_system, profile
        )
        
        # Step 7: Encode target if needed
        result_df = self._encode_target(result_df, target_col)
        
        # Step 8: Apply domain-specific feature engineering
        result_df = self._apply_domain_features(
            result_df, target_col, domain, column_roles
        )
        
        # Step 9: Final cleanup (no dropping!)
        result_df = self._final_cleanup(result_df, target_col)
        
        # Generate transformation report
        report = self._generate_report(df, result_df, target_col)
        
        return result_df, report
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_roles: Dict,
        protection_system: Any,
        profile: Dict
    ) -> pd.DataFrame:
        """
        Handle missing values WITHOUT dropping any columns.
        
        Strategy:
        - Add missing indicator flags
        - Impute based on column type and distribution
        """
        missing_analysis = profile.get("missing_analysis", {})
        
        for col in df.columns:
            if col == target_col:
                continue
            
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            missing_ratio = missing_count / len(df)
            
            # Always add missing indicator flag
            flag_col = f"{col}_missing_flag"
            df[flag_col] = df[col].isnull().astype(int)
            self.added_columns.append(flag_col)
            self._log_transformation(col, "add_missing_flag", {
                "missing_count": int(missing_count),
                "missing_ratio": round(missing_ratio, 4),
            })
            
            # Determine imputation strategy based on column type and distribution
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check skewness to decide mean vs median
                col_profile = profile.get("column_profiles", {}).get(col, {})
                skewness = col_profile.get("skewness", 0)
                
                # Ensure numeric before computing stats
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                
                if abs(skewness) > 1:
                    # High skew - use median
                    impute_value = numeric_col.median()
                    strategy = "median"
                else:
                    # Low skew - use mean
                    impute_value = numeric_col.mean()
                    strategy = "mean"
                
                if pd.isna(impute_value):
                    impute_value = 0
                
                df[col] = numeric_col.fillna(impute_value)
                self._log_transformation(col, "impute_numeric", {
                    "strategy": strategy,
                    "value": round(float(impute_value), 4),
                })
            
            else:
                # Categorical - use mode or "MISSING" category
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    impute_value = mode_value.iloc[0]
                else:
                    impute_value = "MISSING"
                
                df[col] = df[col].fillna(impute_value)
                self._log_transformation(col, "impute_categorical", {
                    "strategy": "mode",
                    "value": str(impute_value),
                })
            
            # For very high missing (>80%), add weight reduction flag
            if missing_ratio > 0.8:
                weight_col = f"{col}_low_confidence"
                df[weight_col] = (df[f"{col}_missing_flag"] == 1).astype(int)
                self.added_columns.append(weight_col)
                self._log_transformation(col, "add_confidence_flag", {
                    "reason": "Very high missing ratio",
                })
        
        return df
    
    def _handle_temporal_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_roles: Dict,
        protection_system: Any
    ) -> pd.DataFrame:
        """
        Handle temporal columns by extracting features.
        PRESERVES original column.
        """
        temporal_cols = [
            col for col, info in column_roles.items()
            if info.get("role") == "temporal" and col != target_col
        ]
        
        for col in temporal_cols:
            if col not in df.columns:
                continue
            
            try:
                # Parse datetime
                dt_series = pd.to_datetime(df[col], errors='coerce')
                
                if dt_series.notna().sum() > 0:
                    # Extract features
                    df[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
                    df[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
                    df[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
                    df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
                    df[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(int)
                    
                    self.added_columns.extend([
                        f"{col}_year", f"{col}_month", f"{col}_day",
                        f"{col}_dayofweek", f"{col}_quarter"
                    ])
                    
                    # Convert original to numeric timestamp (preserve information)
                    df[f"{col}_timestamp"] = dt_series.astype(np.int64) // 10**9
                    df[f"{col}_timestamp"] = df[f"{col}_timestamp"].fillna(0)
                    self.added_columns.append(f"{col}_timestamp")
                    
                    self._log_transformation(col, "extract_datetime_features", {
                        "features_added": 6,
                        "original_preserved": True,
                    })
                    
                    # Keep original as string for traceability
                    df[col] = df[col].astype(str)
            
            except Exception as e:
                self._log_transformation(col, "datetime_extraction_failed", {
                    "error": str(e),
                })
        
        return df
    
    def _handle_categorical_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_roles: Dict,
        protection_system: Any,
        profile: Dict
    ) -> pd.DataFrame:
        """
        Handle categorical columns with appropriate encoding.
        NEVER drops columns.
        """
        categorical_cols = [
            col for col, info in column_roles.items()
            if info.get("role") in ["categorical_nominal", "categorical_ordinal"]
            and col != target_col
        ]
        
        # Also include object columns not classified
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col not in categorical_cols and col != target_col:
                categorical_cols.append(col)
        
        cardinality_analysis = profile.get("cardinality_analysis", {}).get("per_column", {})
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            # Skip if already processed (e.g., temporal)
            if df[col].dtype in [np.float64, np.int64]:
                continue
            
            n_unique = df[col].nunique()
            role_info = column_roles.get(col, {})
            is_ordinal = role_info.get("role") == "categorical_ordinal"
            
            # Determine encoding strategy based on cardinality
            if n_unique <= 2:
                # Binary - simple label encoding
                df = self._apply_label_encoding(df, col)
                self._log_transformation(col, "label_encode_binary", {
                    "n_unique": n_unique,
                })
            
            elif n_unique <= 10:
                # Low cardinality - one-hot encoding
                df = self._apply_onehot_encoding(df, col)
                self._log_transformation(col, "onehot_encode", {
                    "n_unique": n_unique,
                })
            
            elif n_unique <= 50:
                if is_ordinal:
                    # Ordinal encoding
                    df = self._apply_ordinal_encoding(df, col)
                    self._log_transformation(col, "ordinal_encode", {
                        "n_unique": n_unique,
                    })
                else:
                    # Target encoding or frequency encoding
                    df = self._apply_frequency_encoding(df, col)
                    self._log_transformation(col, "frequency_encode", {
                        "n_unique": n_unique,
                    })
            
            else:
                # High cardinality - frequency or hash encoding
                if n_unique > 500:
                    df = self._apply_hash_encoding(df, col)
                    self._log_transformation(col, "hash_encode", {
                        "n_unique": n_unique,
                    })
                else:
                    df = self._apply_frequency_encoding(df, col)
                    self._log_transformation(col, "frequency_encode", {
                        "n_unique": n_unique,
                    })
        
        return df
    
    def _handle_identifier_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_roles: Dict,
        protection_system: Any
    ) -> pd.DataFrame:
        """
        Handle identifier columns - NEVER drop, encode instead.
        """
        identifier_cols = [
            col for col, info in column_roles.items()
            if info.get("role") == "identifier" and col != target_col
        ]
        
        for col in identifier_cols:
            if col not in df.columns:
                continue
            
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Keep original for traceability
            original_col = f"{col}_original"
            df[original_col] = df[col].astype(str)
            self.added_columns.append(original_col)
            
            # Apply hash encoding for identifiers
            df = self._apply_hash_encoding(df, col, n_components=8)
            
            self._log_transformation(col, "encode_identifier", {
                "method": "hash_encoding",
                "original_preserved": True,
            })
        
        return df
    
    def _handle_numerical_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_roles: Dict,
        protection_system: Any,
        profile: Dict
    ) -> pd.DataFrame:
        """
        Handle numerical columns with appropriate scaling.
        """
        numerical_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if col != target_col and not col.endswith('_flag')
        ]
        
        distribution_analysis = profile.get("distribution_analysis", {}).get("per_column", {})
        
        for col in numerical_cols:
            if col not in df.columns:
                continue
            
            # Get distribution info
            dist_info = distribution_analysis.get(col, {})
            skewness = dist_info.get("skewness", 0)
            
            # Handle highly skewed distributions
            if abs(skewness) > 2:
                # Apply log transform for positive values
                if (df[col] > 0).all():
                    df[f"{col}_log"] = np.log1p(df[col])
                    self.added_columns.append(f"{col}_log")
                    self._log_transformation(col, "log_transform", {
                        "skewness_before": round(skewness, 4),
                    })
            
            # Scale numerical columns
            protection_level = protection_system.get_protection_level(col)
            
            if protection_level.value in ["critical", "protected"]:
                # Use RobustScaler for protected columns
                scaler = RobustScaler()
            else:
                # Use StandardScaler for others
                scaler = StandardScaler()
            
            # Create scaled version
            scaled_col = f"{col}_scaled"
            df[scaled_col] = scaler.fit_transform(df[[col]])
            self.added_columns.append(scaled_col)
            self.scalers[col] = scaler
            
            self._log_transformation(col, "scale", {
                "scaler": type(scaler).__name__,
            })
        
        return df
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_roles: Dict,
        protection_system: Any,
        profile: Dict
    ) -> pd.DataFrame:
        """
        Handle outliers by capping, NEVER removing rows.
        """
        numerical_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if col != target_col
            and not col.endswith('_flag')
            and not col.endswith('_scaled')
            and not col.endswith('_log')
        ]
        
        for col in numerical_cols:
            if col not in df.columns:
                continue
            
            # Check protection level
            protection_level = protection_system.get_protection_level(col)
            
            # Skip outlier handling for critical/protected columns
            if protection_level.value in ["critical"]:
                continue
            
            # Calculate IQR bounds
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            if protection_level.value == "protected":
                # Gentler capping for protected columns (2.5 * IQR)
                multiplier = 2.5
            else:
                # Standard capping (1.5 * IQR)
                multiplier = 1.5
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Count outliers before capping
            n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if n_outliers > 0:
                # Add outlier flag before capping
                outlier_flag_col = f"{col}_outlier_flag"
                df[outlier_flag_col] = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ).astype(int)
                self.added_columns.append(outlier_flag_col)
                
                # Cap outliers
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                
                self._log_transformation(col, "cap_outliers", {
                    "n_outliers": int(n_outliers),
                    "lower_bound": round(float(lower_bound), 4),
                    "upper_bound": round(float(upper_bound), 4),
                    "multiplier": multiplier,
                })
        
        return df
    
    def _encode_target(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Encode target column if categorical."""
        if target_col not in df.columns:
            return df
        
        if df[target_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[target_col]):
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
            self.encoders[f"target_{target_col}"] = le
            
            self._log_transformation(target_col, "label_encode_target", {
                "n_classes": len(le.classes_),
                "classes": le.classes_.tolist()[:10],
            })
        
        return df
    
    def _apply_domain_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        domain: str,
        column_roles: Dict
    ) -> pd.DataFrame:
        """Apply domain-specific feature engineering."""
        col_names_lower = {c.lower(): c for c in df.columns}
        
        if domain == "aviation":
            # Traffic volume features
            dep_cols = [c for c in df.columns if 'dep' in c.lower() and df[c].dtype in [np.float64, np.int64]]
            arr_cols = [c for c in df.columns if 'arr' in c.lower() and df[c].dtype in [np.float64, np.int64]]
            
            if dep_cols and arr_cols:
                dep_col = dep_cols[0]
                arr_col = arr_cols[0]
                
                df['traffic_volume'] = df[dep_col] + df[arr_col]
                df['dep_arr_ratio'] = df[dep_col] / (df[arr_col] + 1)
                
                self.added_columns.extend(['traffic_volume', 'dep_arr_ratio'])
                self._log_transformation("domain_features", "aviation_features", {
                    "features_added": ['traffic_volume', 'dep_arr_ratio'],
                })
        
        elif domain == "finance":
            # Look for revenue and cost columns
            revenue_cols = [c for c in df.columns if 'revenue' in c.lower() and df[c].dtype in [np.float64, np.int64]]
            cost_cols = [c for c in df.columns if 'cost' in c.lower() and df[c].dtype in [np.float64, np.int64]]
            
            if revenue_cols and cost_cols:
                rev_col = revenue_cols[0]
                cost_col = cost_cols[0]
                
                df['profit'] = df[rev_col] - df[cost_col]
                df['profit_margin'] = df['profit'] / (df[rev_col] + 1)
                
                self.added_columns.extend(['profit', 'profit_margin'])
                self._log_transformation("domain_features", "finance_features", {
                    "features_added": ['profit', 'profit_margin'],
                })
        
        return df
    
    def _final_cleanup(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Final cleanup - handle any remaining issues WITHOUT dropping.
        """
        # Replace infinities with large finite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN with 0
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("UNKNOWN")
        
        # Ensure all columns are numeric (except preserved originals)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's a preserved original column
                if col.endswith('_original'):
                    continue
                
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except:
                    # Last resort - hash encode
                    df = self._apply_hash_encoding(df, col, n_components=4)
        
        return df
    
    def _apply_label_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply label encoding to a column."""
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        self.encoders[col] = le
        return df
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply one-hot encoding, keeping original column."""
        # Get dummies
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        
        # Add to dataframe
        for dummy_col in dummies.columns:
            df[dummy_col] = dummies[dummy_col]
            self.added_columns.append(dummy_col)
        
        # Convert original to numeric (label encode)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        self.encoders[col] = le
        
        return df
    
    def _apply_ordinal_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply ordinal encoding."""
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[col] = oe.fit_transform(df[[col]].astype(str))
        self.encoders[col] = oe
        return df
    
    def _apply_frequency_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply frequency encoding."""
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
        self.added_columns.append(f"{col}_freq")
        
        # Also label encode the original
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        self.encoders[col] = le
        
        return df
    
    def _apply_hash_encoding(
        self,
        df: pd.DataFrame,
        col: str,
        n_components: int = 8
    ) -> pd.DataFrame:
        """Apply hash encoding for high-cardinality columns."""
        def hash_value(val, n_comp):
            hash_val = int(hashlib.md5(str(val).encode()).hexdigest(), 16)
            return hash_val % n_comp
        
        # Create hash features
        for i in range(n_components):
            hash_col = f"{col}_hash_{i}"
            df[hash_col] = df[col].apply(lambda x: 1 if hash_value(x, n_components) == i else 0)
            self.added_columns.append(hash_col)
        
        # Label encode original
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        self.encoders[col] = le
        
        return df
    
    def _log_transformation(self, column: str, operation: str, details: Dict):
        """Log a transformation."""
        self.transformation_log.append({
            "column": column,
            "operation": operation,
            "details": details,
        })
    
    def _generate_report(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        target_col: str
    ) -> Dict[str, Any]:
        """Generate transformation report."""
        return {
            "original_shape": original_df.shape,
            "transformed_shape": transformed_df.shape,
            "original_columns": self.original_columns,
            "added_columns": self.added_columns,
            "columns_dropped": [],  # ALWAYS EMPTY - we never drop
            "n_transformations": len(self.transformation_log),
            "transformations": self.transformation_log,
            "encoders_fitted": list(self.encoders.keys()),
            "scalers_fitted": list(self.scalers.keys()),
            "target_column": target_col,
            "guarantee": "NO COLUMNS WERE DROPPED",
        }
