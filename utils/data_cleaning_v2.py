"""
ADIE — Data Cleaning v2.0
==========================

This module provides the new adaptive cleaning interface while maintaining
full backward compatibility with the existing ADIE system.

The new system GUARANTEES:
1. NO COLUMN IS EVER DROPPED
2. All transformations are traceable
3. Domain context drives decisions
4. Feature importance protects critical columns

Usage:
    # New adaptive interface (recommended)
    from utils.data_cleaning_v2 import clean_dataset_v2
    cleaned_df, report = clean_dataset_v2(df, target_col)
    
    # Backward compatible interface
    from utils.data_cleaning_v2 import clean_dataset
    cleaned_df = clean_dataset(df, target_col=target_col)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the new adaptive engine
try:
    from utils.adaptive_engine import AdaptiveDataPreparationEngine
    from utils.adaptive_cleaning import (
        clean_dataset_adaptive,
        get_adaptive_diagnostics
    )
    ADAPTIVE_ENGINE_AVAILABLE = True
except ImportError as e:
    ADAPTIVE_ENGINE_AVAILABLE = False
    print(f"Warning: Adaptive engine not available: {e}")

# Import legacy cleaning as fallback
from utils.data_cleaning import clean_dataset as legacy_clean_dataset


def clean_dataset_v2(
    df: pd.DataFrame,
    target_col: str,
    leakage_cols: Optional[List[str]] = None,
    use_adaptive: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset using the new Adaptive Data Preparation Engine.
    
    This is the recommended interface for new code.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    leakage_cols : list, optional
        Columns suspected of data leakage (handled specially, NOT dropped)
    use_adaptive : bool
        Use adaptive engine (True) or legacy cleaning (False)
    verbose : bool
        Enable verbose logging
        
    Returns
    -------
    (cleaned_df, preparation_report)
    """
    if use_adaptive and ADAPTIVE_ENGINE_AVAILABLE:
        return clean_dataset_adaptive(
            df, target_col, leakage_cols,
            config={"verbose": verbose},
            verbose=verbose
        )
    else:
        # Fallback to legacy cleaning
        cleaned_df = legacy_clean_dataset(
            df, leakage_cols=leakage_cols, target_col=target_col
        )
        report = {
            "version": "1.0 (legacy)",
            "method": "legacy_clean_dataset",
            "adaptive_available": ADAPTIVE_ENGINE_AVAILABLE,
        }
        return cleaned_df, report


def clean_dataset(
    df: pd.DataFrame,
    leakage_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    fit_encoders: bool = True
) -> pd.DataFrame:
    """
    Backward-compatible clean_dataset function.
    
    This maintains the exact same signature as the original function
    for seamless integration with existing code.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    leakage_cols : list, optional
        Columns suspected of data leakage
    target_col : str, optional
        Target column name (auto-detect if None)
    fit_encoders : bool
        Whether to fit new encoders (parameter kept for compatibility)
        
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
    
    # Try adaptive cleaning first
    if ADAPTIVE_ENGINE_AVAILABLE:
        try:
            cleaned_df, _ = clean_dataset_adaptive(
                df, target_col, leakage_cols, verbose=False
            )
            return cleaned_df
        except Exception as e:
            print(f"Warning: Adaptive cleaning failed, falling back to legacy: {e}")
    
    # Fallback to legacy cleaning
    return legacy_clean_dataset(
        df, leakage_cols=leakage_cols, target_col=target_col, fit_encoders=fit_encoders
    )


def get_cleaning_strategy(
    df: pd.DataFrame,
    target_col: str
) -> Dict[str, Any]:
    """
    Get the cleaning strategy that would be applied without actually cleaning.
    
    Useful for previewing what transformations will be applied.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
        
    Returns
    -------
    dict
        Strategy information including:
        - dataset_type
        - domain
        - protected_columns
        - transformations planned
        - encoding strategy
        - imputation strategy
    """
    if not ADAPTIVE_ENGINE_AVAILABLE:
        return {
            "error": "Adaptive engine not available",
            "fallback": "legacy_cleaning",
        }
    
    try:
        # Initialize engine but don't run full preparation
        engine = AdaptiveDataPreparationEngine({"verbose": False})
        
        # Run profiling and detection only
        df = df.copy()
        df.columns = df.columns.str.strip()
        target_col = target_col.strip()
        
        profile = engine.profiler.profile_dataset(df, target_col)
        type_info = engine.type_detector.detect_type(df, profile)
        domain_info = engine.domain_detector.detect_domain(df, profile)
        column_roles = engine.role_classifier.classify_columns(df, target_col, profile, domain_info)
        importance = engine.importance_analyzer.analyze_importance(df, target_col, profile)
        
        return {
            "dataset_type": type_info["primary_type"],
            "dataset_type_confidence": type_info["confidence"],
            "domain": domain_info["domain"],
            "domain_confidence": domain_info["confidence"],
            "protected_columns": importance.get("protected_columns", []),
            "careful_columns": importance.get("careful_columns", []),
            "column_roles": {
                col: info["role"] for col, info in column_roles.items()
            },
            "recommendations": type_info.get("recommendations", []),
            "feature_engineering_suggestions": domain_info.get("feature_engineering_suggestions", []),
            "quality_score": profile.get("quality_metrics", {}).get("overall_quality_score", 0),
        }
    except Exception as e:
        return {
            "error": str(e),
            "fallback": "legacy_cleaning",
        }


# Export all functions
__all__ = [
    'clean_dataset',
    'clean_dataset_v2',
    'get_cleaning_strategy',
    'ADAPTIVE_ENGINE_AVAILABLE',
]
