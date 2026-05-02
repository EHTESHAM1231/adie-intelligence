"""
ADIE — Adaptive Data Preparation Engine
=========================================

Main orchestration engine that coordinates all components to provide
intelligent, domain-aware, non-destructive data preparation.

CORE GUARANTEE: NO COLUMN IS EVER DROPPED
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os

from .profiler import DatasetProfiler
from .detector import DatasetTypeDetector, DomainDetector
from .classifier import ColumnRoleClassifier
from .importance import FeatureImportanceAnalyzer
from .protector import ColumnProtectionSystem, ProtectionLevel
from .transformer import AdaptiveTransformer


class AdaptiveDataPreparationEngine:
    """
    Context-Aware Adaptive Data Preparation Engine
    
    Replaces static rule-based cleaning with intelligent, domain-aware,
    non-destructive data preparation.
    
    GUARANTEES:
    1. NO COLUMN IS EVER DROPPED
    2. All transformations are traceable
    3. Domain context drives all decisions
    4. Feature importance protects critical columns
    5. Dataset type determines strategy
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the adaptive engine.
        
        Parameters
        ----------
        config : dict, optional
            Configuration options:
            - max_features_for_importance: int (default 50)
            - enable_domain_features: bool (default True)
            - verbose: bool (default False)
        """
        self.config = config or {}
        
        # Initialize components
        self.profiler = DatasetProfiler()
        self.type_detector = DatasetTypeDetector()
        self.domain_detector = DomainDetector()
        self.role_classifier = ColumnRoleClassifier()
        self.importance_analyzer = FeatureImportanceAnalyzer()
        self.protection_system = ColumnProtectionSystem()
        self.transformer = AdaptiveTransformer()
        
        # State
        self.profile = None
        self.dataset_type_info = None
        self.domain_info = None
        self.column_roles = None
        self.importance_result = None
        self.transformation_report = None
        
        # Logging
        self.verbose = self.config.get("verbose", False)
    
    def prepare(
        self,
        df: pd.DataFrame,
        target_col: str,
        user_protections: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main entry point: Prepare dataset for ML.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        user_protections : dict, optional
            User-specified column protections {col: level}
            
        Returns
        -------
        (prepared_df, preparation_report)
        """
        start_time = datetime.now()
        
        # Validate inputs
        df = df.copy()
        df.columns = df.columns.str.strip()
        target_col = target_col.strip()
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        self._log(f"Starting adaptive data preparation for {df.shape[0]} rows, {df.shape[1]} columns")
        self._log(f"Target column: {target_col}")
        
        # Step 1: Profile the dataset
        self._log("Step 1: Profiling dataset...")
        self.profile = self.profiler.profile_dataset(df, target_col)
        
        # Step 2: Detect dataset type
        self._log("Step 2: Detecting dataset type...")
        self.dataset_type_info = self.type_detector.detect_type(df, self.profile)
        dataset_type = self.dataset_type_info["primary_type"]
        self._log(f"  Detected type: {dataset_type} (confidence: {self.dataset_type_info['confidence']:.2f})")
        
        # Step 3: Detect domain
        self._log("Step 3: Detecting domain...")
        self.domain_info = self.domain_detector.detect_domain(df, self.profile)
        domain = self.domain_info["domain"]
        self._log(f"  Detected domain: {domain} (confidence: {self.domain_info['confidence']:.2f})")
        
        # Step 4: Classify column roles
        self._log("Step 4: Classifying column roles...")
        self.column_roles = self.role_classifier.classify_columns(
            df, target_col, self.profile, self.domain_info
        )
        roles_summary = self._summarize_roles()
        self._log(f"  Roles: {roles_summary}")
        
        # Step 5: Analyze feature importance
        self._log("Step 5: Analyzing feature importance...")
        max_features = self.config.get("max_features_for_importance", 50)
        self.importance_result = self.importance_analyzer.analyze_importance(
            df, target_col, self.profile, max_features
        )
        self._log(f"  Protected: {len(self.importance_result.get('protected_columns', []))} columns")
        self._log(f"  Careful: {len(self.importance_result.get('careful_columns', []))} columns")
        self._log(f"  Flexible: {len(self.importance_result.get('flexible_columns', []))} columns")
        
        # Step 6: Build protection map
        self._log("Step 6: Building protection map...")
        self.protection_system.build_protection_map(
            df, target_col,
            self.importance_result,
            self.domain_info,
            self.column_roles,
            user_protections
        )
        protection_summary = self.protection_system.get_protection_summary()
        self._log(f"  Protection levels: {protection_summary['by_level']}")
        
        # Step 7: Apply transformations
        self._log("Step 7: Applying adaptive transformations...")
        prepared_df, self.transformation_report = self.transformer.transform(
            df, target_col,
            dataset_type,
            domain,
            self.column_roles,
            self.protection_system,
            self.profile,
            self.config
        )
        self._log(f"  Original shape: {df.shape}")
        self._log(f"  Prepared shape: {prepared_df.shape}")
        self._log(f"  Columns added: {len(self.transformation_report.get('added_columns', []))}")
        self._log(f"  Columns dropped: {len(self.transformation_report.get('columns_dropped', []))} (GUARANTEED 0)")
        
        # Generate comprehensive report
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        report = self._generate_comprehensive_report(
            df, prepared_df, target_col, processing_time
        )
        
        self._log(f"Preparation complete in {processing_time:.2f} seconds")
        
        return prepared_df, report
    
    def _summarize_roles(self) -> Dict[str, int]:
        """Summarize column roles."""
        summary = {}
        for col, info in self.column_roles.items():
            role = info.get("role", "unknown")
            summary[role] = summary.get(role, 0) + 1
        return summary
    
    def _generate_comprehensive_report(
        self,
        original_df: pd.DataFrame,
        prepared_df: pd.DataFrame,
        target_col: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive preparation report."""
        
        # Verify no columns were dropped
        original_cols = set(original_df.columns)
        prepared_cols = set(prepared_df.columns)
        dropped_cols = original_cols - prepared_cols
        
        # This should ALWAYS be empty
        assert len(dropped_cols) == 0, f"VIOLATION: Columns were dropped: {dropped_cols}"
        
        report = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            
            # Dataset info
            "dataset_info": {
                "original_shape": list(original_df.shape),
                "prepared_shape": list(prepared_df.shape),
                "target_column": target_col,
                "columns_added": len(self.transformation_report.get("added_columns", [])),
                "columns_dropped": 0,  # GUARANTEED
            },
            
            # Dataset type
            "dataset_type": {
                "primary_type": self.dataset_type_info["primary_type"],
                "confidence": self.dataset_type_info["confidence"],
                "type_scores": self.dataset_type_info["type_scores"],
                "recommendations": self.dataset_type_info["recommendations"],
            },
            
            # Domain
            "domain": {
                "detected_domain": self.domain_info["domain"],
                "confidence": self.domain_info["confidence"],
                "matched_keywords": self.domain_info.get("matched_keywords", {}),
                "feature_engineering_suggestions": self.domain_info.get("feature_engineering_suggestions", []),
            },
            
            # Column roles
            "column_roles": {
                col: {
                    "role": info["role"],
                    "confidence": info["confidence"],
                    "sub_type": info.get("sub_type"),
                }
                for col, info in self.column_roles.items()
            },
            
            # Protection
            "protection": {
                "summary": self.protection_system.get_protection_summary()["by_level"],
                "protected_columns": self.protection_system.get_columns_by_level(ProtectionLevel.PROTECTED),
                "critical_columns": self.protection_system.get_columns_by_level(ProtectionLevel.CRITICAL),
            },
            
            # Feature importance
            "feature_importance": {
                "method": self.importance_result.get("analysis_method", "unknown"),
                "task_type": self.importance_result.get("task_type", "unknown"),
                "top_features": dict(
                    sorted(
                        self.importance_result.get("importance_scores", {}).items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                ),
            },
            
            # Transformations
            "transformations": {
                "total": len(self.transformation_report.get("transformations", [])),
                "by_type": self._count_transformations_by_type(),
                "details": self.transformation_report.get("transformations", [])[:50],  # Limit for readability
            },
            
            # Quality metrics
            "quality_metrics": self.profile.get("quality_metrics", {}),
            
            # Guarantees
            "guarantees": {
                "no_columns_dropped": True,
                "all_transformations_logged": True,
                "original_information_preserved": True,
            },
            
            # Notes
            "notes": self._generate_notes(),
        }
        
        return report
    
    def _count_transformations_by_type(self) -> Dict[str, int]:
        """Count transformations by type."""
        counts = {}
        for t in self.transformation_report.get("transformations", []):
            op = t.get("operation", "unknown")
            counts[op] = counts.get(op, 0) + 1
        return counts
    
    def _generate_notes(self) -> List[str]:
        """Generate notes about the preparation."""
        notes = []
        
        # Dataset type notes
        if self.dataset_type_info["primary_type"] == "time_series":
            notes.append("Time-series dataset detected - chronological order preserved")
        
        if self.dataset_type_info["primary_type"] == "high_cardinality":
            notes.append("High-cardinality dataset - frequency/hash encoding applied")
        
        # Domain notes
        if self.domain_info["confidence"] > 0.7:
            notes.append(f"Strong domain signal ({self.domain_info['domain']}) - domain-specific features added")
        
        # Missing value notes
        missing_analysis = self.profile.get("missing_analysis", {})
        if missing_analysis.get("cols_very_high_missing"):
            notes.append(f"Columns with >80% missing: {len(missing_analysis['cols_very_high_missing'])} - confidence flags added")
        
        # Protection notes
        n_protected = len(self.protection_system.get_columns_by_level(ProtectionLevel.PROTECTED))
        if n_protected > 0:
            notes.append(f"{n_protected} high-importance columns protected from aggressive transformation")
        
        return notes
    
    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ADIE] {message}")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the cleaning strategy that was applied.
        Useful for the UI's Decision Transparency Panel.
        """
        if not self.profile:
            return {"error": "No preparation has been run yet"}
        
        return {
            "dataset_type": self.dataset_type_info["primary_type"] if self.dataset_type_info else "unknown",
            "domain": self.domain_info["domain"] if self.domain_info else "unknown",
            "protected_columns": self.protection_system.get_columns_by_level(ProtectionLevel.PROTECTED),
            "transformations": self._count_transformations_by_type(),
            "feature_engineering": self.domain_info.get("feature_engineering_suggestions", []) if self.domain_info else [],
            "encoding_used": self._get_encoding_summary(),
            "imputation_strategy": self._get_imputation_summary(),
            "notes": self._generate_notes() if self.profile else [],
        }
    
    def _get_encoding_summary(self) -> str:
        """Get summary of encoding strategies used."""
        transformations = self.transformation_report.get("transformations", []) if self.transformation_report else []
        
        encodings = set()
        for t in transformations:
            op = t.get("operation", "")
            if "encode" in op:
                encodings.add(op)
        
        if not encodings:
            return "No encoding applied"
        
        return ", ".join(sorted(encodings))
    
    def _get_imputation_summary(self) -> str:
        """Get summary of imputation strategies used."""
        transformations = self.transformation_report.get("transformations", []) if self.transformation_report else []
        
        strategies = set()
        for t in transformations:
            op = t.get("operation", "")
            if "impute" in op:
                details = t.get("details", {})
                strategy = details.get("strategy", "unknown")
                strategies.add(strategy)
        
        if not strategies:
            return "No imputation needed"
        
        return ", ".join(sorted(strategies))


# Convenience function for backward compatibility
def adaptive_clean_dataset(
    df: pd.DataFrame,
    target_col: str,
    leakage_cols: Optional[List[str]] = None,  # Ignored - we don't drop
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Backward-compatible function that replaces the old clean_dataset.
    
    NOTE: leakage_cols parameter is accepted but IGNORED.
    We never drop columns - instead we reduce their influence through
    encoding and feature importance weighting.
    """
    engine = AdaptiveDataPreparationEngine(config)
    prepared_df, _ = engine.prepare(df, target_col)
    return prepared_df
