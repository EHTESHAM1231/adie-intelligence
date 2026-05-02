"""
ADIE — Adaptive Cleaning Integration
=====================================

Integration layer that connects the new Adaptive Data Preparation Engine
with the existing ADIE system, providing backward compatibility while
enabling the new intelligent, non-destructive cleaning capabilities.

This module replaces the old data_cleaning.py functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import joblib

# Import the new adaptive engine
from utils.adaptive_engine import (
    AdaptiveDataPreparationEngine,
    DatasetProfiler,
    DatasetTypeDetector,
    DomainDetector,
    ColumnRoleClassifier,
    FeatureImportanceAnalyzer,
    ColumnProtectionSystem,
    AdaptiveTransformer
)

# Paths for saving artifacts
UPLOAD_FOLDER = 'uploads'
ENCODER_PATH = os.path.join(UPLOAD_FOLDER, 'encoder_mappings.pkl')
ADAPTIVE_REPORT_PATH = os.path.join(UPLOAD_FOLDER, 'adaptive_report.json')


def clean_dataset_adaptive(
    df: pd.DataFrame,
    target_col: str,
    leakage_cols: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset using the new Adaptive Data Preparation Engine.
    
    This is the main entry point that replaces the old clean_dataset function.
    
    GUARANTEES:
    - NO COLUMN IS EVER DROPPED
    - All transformations are traceable
    - Domain context drives decisions
    - Feature importance protects critical columns
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    leakage_cols : list, optional
        Columns suspected of data leakage (handled specially, NOT dropped)
    config : dict, optional
        Configuration options
    verbose : bool
        Enable verbose logging
        
    Returns
    -------
    (cleaned_df, preparation_report)
    """
    # Initialize engine with config
    engine_config = config or {}
    engine_config["verbose"] = verbose
    
    engine = AdaptiveDataPreparationEngine(engine_config)
    
    # Handle leakage columns through user protections
    # Instead of dropping, we mark them for special handling
    user_protections = {}
    if leakage_cols:
        for col in leakage_cols:
            if col != target_col:
                # Mark as structural - will be encoded but not used directly
                user_protections[col] = "structural"
    
    # Run adaptive preparation
    prepared_df, report = engine.prepare(df, target_col, user_protections)
    
    # Save artifacts
    _save_artifacts(engine, report)
    
    return prepared_df, report


def clean_dataset(
    df: pd.DataFrame,
    leakage_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    fit_encoders: bool = True
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for the old clean_dataset function.
    
    This function maintains the same signature as the original but uses
    the new adaptive engine internally.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    leakage_cols : list, optional
        Columns suspected of data leakage
    target_col : str, optional
        Target column name (auto-detect if None)
    fit_encoders : bool
        Whether to fit new encoders (ignored - always fits)
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    # Sanitize inputs
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Auto-detect target if not provided
    if target_col is None:
        target_col = df.columns[-1]
    else:
        target_col = target_col.strip()
    
    # Verify target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    # Use adaptive cleaning
    prepared_df, report = clean_dataset_adaptive(
        df, target_col, leakage_cols, verbose=False
    )
    
    return prepared_df


def get_adaptive_diagnostics(
    df: pd.DataFrame,
    target_col: str
) -> Dict[str, Any]:
    """
    Get comprehensive diagnostics using the adaptive engine components.
    
    This provides richer diagnostics than the original perform_diagnostics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
        
    Returns
    -------
    dict
        Comprehensive diagnostics including:
        - Dataset profile
        - Dataset type detection
        - Domain detection
        - Column role classification
        - Feature importance analysis
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    target_col = target_col.strip()
    
    # Profile dataset
    profiler = DatasetProfiler()
    profile = profiler.profile_dataset(df, target_col)
    
    # Detect dataset type
    type_detector = DatasetTypeDetector()
    type_info = type_detector.detect_type(df, profile)
    
    # Detect domain
    domain_detector = DomainDetector()
    domain_info = domain_detector.detect_domain(df, profile)
    
    # Classify column roles
    role_classifier = ColumnRoleClassifier()
    column_roles = role_classifier.classify_columns(df, target_col, profile, domain_info)
    
    # Analyze feature importance
    importance_analyzer = FeatureImportanceAnalyzer()
    importance_result = importance_analyzer.analyze_importance(df, target_col, profile)
    
    # Build comprehensive diagnostics
    diagnostics = {
        # Basic info
        "rows": profile["n_rows"],
        "columns": profile["n_cols"],
        "size_class": profile["size_class"],
        "memory_mb": profile["memory_mb"],
        
        # Missing values (compatible with old format)
        "missing_values": {
            "total": profile["missing_analysis"]["total_missing"],
            "by_column": profile["missing_analysis"]["missing_per_column"],
        },
        
        # Duplicates
        "duplicates": profile["duplicate_analysis"]["n_duplicate_rows"],
        
        # Outliers
        "outliers": _extract_outlier_info(profile),
        
        # Class imbalance
        "class_imbalance": _extract_class_imbalance(profile, target_col),
        
        # Correlations
        "correlations": _extract_correlations(profile),
        
        # Leakage risk (high correlations with target)
        "leakage_risk": _identify_leakage_risk(profile, target_col),
        
        # Label noise (estimated)
        "label_noise": 0,  # Would need KNN analysis
        
        # Mixed fields
        "mixed_fields": _identify_mixed_fields(df),
        
        # Identified issues (compatible with old format)
        "identified_issues": _identify_issues(profile, target_col),
        
        # Column types (compatible with old format)
        "column_types": _extract_column_types(column_roles),
        
        # NEW: Enhanced diagnostics
        "enhanced": {
            "dataset_type": type_info,
            "domain": domain_info,
            "column_roles": column_roles,
            "feature_importance": importance_result,
            "quality_metrics": profile["quality_metrics"],
            "structural_patterns": profile["structural_patterns"],
        },
        
        # Target column
        "target_col": target_col,
    }
    
    return diagnostics


def _extract_outlier_info(profile: Dict) -> Dict:
    """Extract outlier information from profile."""
    total_outliers = 0
    by_column = {}
    
    for col, col_profile in profile.get("column_profiles", {}).items():
        n_outliers = col_profile.get("n_outliers", 0)
        if n_outliers > 0:
            by_column[col] = n_outliers
            total_outliers += n_outliers
    
    return {
        "total": total_outliers,
        "by_column": by_column,
    }


def _extract_class_imbalance(profile: Dict, target_col: str) -> Dict:
    """Extract class imbalance information."""
    target_analysis = profile.get("target_analysis", {})
    
    return {
        "target_column": target_col,
        "distribution": target_analysis.get("class_distribution", {}),
    }


def _extract_correlations(profile: Dict) -> Dict:
    """Extract top correlations."""
    corr_analysis = profile.get("correlation_analysis", {})
    high_corrs = corr_analysis.get("high_correlations", [])
    
    result = {}
    for corr in high_corrs[:5]:
        key = f"{corr['col1']}_vs_{corr['col2']}"
        result[key] = corr["correlation"]
    
    return result


def _identify_leakage_risk(profile: Dict, target_col: str) -> List[str]:
    """Identify columns with potential data leakage."""
    leakage_cols = []
    
    corr_analysis = profile.get("correlation_analysis", {})
    corr_matrix = corr_analysis.get("correlation_matrix", {})
    
    if target_col in corr_matrix:
        target_corrs = corr_matrix[target_col]
        for col, corr in target_corrs.items():
            if col != target_col and abs(corr) > 0.95:
                leakage_cols.append(col)
    
    return leakage_cols


def _identify_mixed_fields(df: pd.DataFrame) -> Dict:
    """Identify columns with mixed data types."""
    mixed_fields = {}
    
    for col in df.select_dtypes(include='object').columns:
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
        
        # Check for mixed numeric/text
        numeric_mask = pd.to_numeric(sample, errors='coerce').notna()
        numeric_ratio = numeric_mask.sum() / len(sample)
        
        if 0.1 < numeric_ratio < 0.9:
            mixed_fields[col] = {
                "type": "Mixed numeric and text",
                "numeric_count": int(numeric_mask.sum()),
                "text_count": int((~numeric_mask).sum()),
            }
    
    return mixed_fields


def _identify_issues(profile: Dict, target_col: str) -> List[Dict]:
    """Identify data quality issues (compatible with old format)."""
    issues = []
    
    n_rows = profile["n_rows"]
    n_cols = profile["n_cols"]
    
    # Missing values
    missing_analysis = profile.get("missing_analysis", {})
    missing_ratio = missing_analysis.get("overall_missing_ratio", 0)
    if missing_ratio > 0.05:
        issues.append({
            "type": "Missing Values",
            "severity": "High" if missing_ratio > 0.2 else "Medium",
            "score": missing_ratio,
        })
    
    # Duplicates
    dup_analysis = profile.get("duplicate_analysis", {})
    dup_ratio = dup_analysis.get("duplicate_ratio", 0)
    if dup_ratio > 0.05:
        issues.append({
            "type": "Redundancy",
            "severity": "High" if dup_ratio > 0.15 else "Medium",
            "score": dup_ratio,
        })
    
    # Class imbalance
    target_analysis = profile.get("target_analysis", {})
    imbalance_ratio = target_analysis.get("imbalance_ratio", 1)
    if imbalance_ratio > 5:
        issues.append({
            "type": "Class Imbalance",
            "severity": "High" if imbalance_ratio > 20 else "Medium",
            "score": imbalance_ratio,
        })
    
    # Outliers
    outlier_info = _extract_outlier_info(profile)
    total_outliers = outlier_info["total"]
    numeric_cols = len(profile.get("numeric_cols", []))
    if numeric_cols > 0:
        outlier_ratio = total_outliers / (n_rows * numeric_cols)
        if outlier_ratio > 0.1:
            issues.append({
                "type": "Outliers",
                "severity": "High" if outlier_ratio > 0.25 else "Medium",
                "score": outlier_ratio,
            })
    
    # High correlation (potential leakage)
    leakage_cols = _identify_leakage_risk(profile, target_col)
    if leakage_cols:
        issues.append({
            "type": "Data Leakage",
            "severity": "High",
            "score": len(leakage_cols),
        })
    
    return issues


def _extract_column_types(column_roles: Dict) -> Dict:
    """Extract column types in old format."""
    types = {
        "identifiers": [],
        "datetime_cols": [],
        "numerical_cols": [],
        "nominal_categorical": [],
        "ordinal_categorical": [],
    }
    
    for col, info in column_roles.items():
        role = info.get("role", "")
        
        if role == "identifier":
            types["identifiers"].append(col)
        elif role == "temporal":
            types["datetime_cols"].append(col)
        elif role in ["numerical_continuous", "numerical_discrete"]:
            types["numerical_cols"].append(col)
        elif role == "categorical_nominal":
            types["nominal_categorical"].append(col)
        elif role == "categorical_ordinal":
            types["ordinal_categorical"].append(col)
    
    return types


def _save_artifacts(engine: AdaptiveDataPreparationEngine, report: Dict):
    """Save transformation artifacts."""
    import json
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Save encoder mappings
    if engine.transformer.encoders:
        joblib.dump(engine.transformer.encoders, ENCODER_PATH)
    
    # Save report (JSON-serializable parts)
    try:
        report_path = ADAPTIVE_REPORT_PATH
        with open(report_path, 'w') as f:
            # Filter out non-serializable parts
            serializable_report = _make_serializable(report)
            json.dump(serializable_report, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save adaptive report: {e}")


def _make_serializable(obj):
    """Make object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


# Export for backward compatibility
__all__ = [
    'clean_dataset',
    'clean_dataset_adaptive',
    'get_adaptive_diagnostics',
    'AdaptiveDataPreparationEngine',
]
