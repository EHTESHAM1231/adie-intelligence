"""
ADIE — Column Protection System
================================

Unified protection system that ensures critical columns are never
dropped or heavily distorted during data preparation.

CORE PRINCIPLE: NO COLUMN IS EVER DROPPED
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class ProtectionLevel(Enum):
    """Protection levels for columns."""
    CRITICAL = "critical"      # Cannot be modified at all (target)
    PROTECTED = "protected"    # High importance - minimal transformation
    CAREFUL = "careful"        # Medium importance - careful transformation
    FLEXIBLE = "flexible"      # Low importance - can transform freely
    STRUCTURAL = "structural"  # Identifiers, keys - preserve but can encode


@dataclass
class ColumnProtection:
    """Protection configuration for a single column."""
    column: str
    level: ProtectionLevel
    reasons: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=list)
    forbidden_operations: List[str] = field(default_factory=list)
    transformation_constraints: Dict[str, Any] = field(default_factory=dict)


class ColumnProtectionSystem:
    """
    Unified protection system that aggregates protection requirements
    from multiple sources and enforces them during transformation.
    
    Protection sources:
    1. Target column (always CRITICAL)
    2. Feature importance analysis
    3. Domain-specific rules
    4. Column role classification
    5. User-specified protections
    """
    
    # Operations that can be applied at each protection level
    ALLOWED_OPERATIONS = {
        ProtectionLevel.CRITICAL: [
            "label_encode_target",  # Only for categorical targets
        ],
        ProtectionLevel.PROTECTED: [
            "impute_missing",
            "add_missing_flag",
            "scale",
            "clip_outliers_gentle",  # Less aggressive clipping
        ],
        ProtectionLevel.CAREFUL: [
            "impute_missing",
            "add_missing_flag",
            "scale",
            "clip_outliers",
            "encode_categorical",
            "extract_datetime",
        ],
        ProtectionLevel.FLEXIBLE: [
            "impute_missing",
            "add_missing_flag",
            "scale",
            "clip_outliers",
            "encode_categorical",
            "extract_datetime",
            "bin_numeric",
            "transform_skewed",
            "hash_encode",
        ],
        ProtectionLevel.STRUCTURAL: [
            "label_encode",
            "hash_encode",
            "frequency_encode",
            "add_missing_flag",
        ],
    }
    
    # Operations that are NEVER allowed (would drop information)
    FORBIDDEN_OPERATIONS = [
        "drop_column",
        "drop_rows_with_missing",
        "remove_outliers",
        "delete_duplicates_aggressive",
    ]
    
    def __init__(self):
        self.protections: Dict[str, ColumnProtection] = {}
        self.protection_log: List[Dict] = []
    
    def build_protection_map(
        self,
        df: pd.DataFrame,
        target_col: str,
        importance_result: Optional[Dict] = None,
        domain_info: Optional[Dict] = None,
        column_roles: Optional[Dict] = None,
        user_protections: Optional[Dict[str, str]] = None
    ) -> Dict[str, ColumnProtection]:
        """
        Build comprehensive protection map for all columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        importance_result : dict, optional
            Result from FeatureImportanceAnalyzer
        domain_info : dict, optional
            Result from DomainDetector
        column_roles : dict, optional
            Result from ColumnRoleClassifier
        user_protections : dict, optional
            User-specified protection levels {col: level}
            
        Returns
        -------
        Dict[str, ColumnProtection]
            Protection configuration for each column
        """
        df.columns = df.columns.str.strip()
        target_col = target_col.strip()
        
        self.protections = {}
        
        # Initialize all columns with FLEXIBLE level
        for col in df.columns:
            self.protections[col] = ColumnProtection(
                column=col,
                level=ProtectionLevel.FLEXIBLE,
                reasons=["Default level"],
                allowed_operations=self.ALLOWED_OPERATIONS[ProtectionLevel.FLEXIBLE].copy(),
                forbidden_operations=self.FORBIDDEN_OPERATIONS.copy(),
            )
        
        # 1. Target column is always CRITICAL
        if target_col in self.protections:
            self._set_protection(
                target_col,
                ProtectionLevel.CRITICAL,
                "Target column - must be preserved exactly"
            )
        
        # 2. Apply importance-based protections
        if importance_result:
            self._apply_importance_protections(importance_result)
        
        # 3. Apply domain-based protections
        if domain_info:
            self._apply_domain_protections(domain_info)
        
        # 4. Apply role-based protections
        if column_roles:
            self._apply_role_protections(column_roles)
        
        # 5. Apply user-specified protections (highest priority)
        if user_protections:
            self._apply_user_protections(user_protections)
        
        # Update allowed operations based on final protection levels
        self._update_allowed_operations()
        
        return self.protections
    
    def _set_protection(
        self,
        col: str,
        level: ProtectionLevel,
        reason: str,
        upgrade_only: bool = True
    ):
        """
        Set protection level for a column.
        
        Parameters
        ----------
        col : str
            Column name
        level : ProtectionLevel
            New protection level
        reason : str
            Reason for this protection
        upgrade_only : bool
            If True, only upgrade protection (never downgrade)
        """
        if col not in self.protections:
            return
        
        current = self.protections[col]
        
        # Protection level hierarchy (higher = more protected)
        hierarchy = {
            ProtectionLevel.FLEXIBLE: 0,
            ProtectionLevel.CAREFUL: 1,
            ProtectionLevel.STRUCTURAL: 2,
            ProtectionLevel.PROTECTED: 3,
            ProtectionLevel.CRITICAL: 4,
        }
        
        if upgrade_only:
            if hierarchy[level] > hierarchy[current.level]:
                current.level = level
                current.reasons.append(reason)
        else:
            current.level = level
            current.reasons.append(reason)
        
        # Log the protection change
        self.protection_log.append({
            "column": col,
            "level": level.value,
            "reason": reason,
        })
    
    def _apply_importance_protections(self, importance_result: Dict):
        """Apply protections based on feature importance."""
        # Protected columns (high importance)
        for col in importance_result.get("protected_columns", []):
            self._set_protection(
                col,
                ProtectionLevel.PROTECTED,
                f"High feature importance ({importance_result['importance_scores'].get(col, 0):.4f})"
            )
        
        # Careful columns (medium importance)
        for col in importance_result.get("careful_columns", []):
            self._set_protection(
                col,
                ProtectionLevel.CAREFUL,
                f"Medium feature importance ({importance_result['importance_scores'].get(col, 0):.4f})"
            )
    
    def _apply_domain_protections(self, domain_info: Dict):
        """Apply protections based on domain context."""
        domain = domain_info.get("domain", "general")
        protected_cols = domain_info.get("protected_columns", [])
        
        for col in protected_cols:
            # Find matching column (case-insensitive)
            for actual_col in self.protections.keys():
                if actual_col.lower() == col.lower():
                    self._set_protection(
                        actual_col,
                        ProtectionLevel.PROTECTED,
                        f"Domain-critical column for {domain}"
                    )
                    break
    
    def _apply_role_protections(self, column_roles: Dict):
        """Apply protections based on column roles."""
        for col, role_info in column_roles.items():
            role = role_info.get("role", "")
            
            if role == "target":
                self._set_protection(
                    col,
                    ProtectionLevel.CRITICAL,
                    "Classified as target column"
                )
            
            elif role == "identifier":
                self._set_protection(
                    col,
                    ProtectionLevel.STRUCTURAL,
                    "Classified as identifier - preserve for traceability"
                )
            
            elif role == "temporal":
                self._set_protection(
                    col,
                    ProtectionLevel.CAREFUL,
                    "Classified as temporal - preserve time information"
                )
            
            elif role == "geographical":
                self._set_protection(
                    col,
                    ProtectionLevel.CAREFUL,
                    "Classified as geographical - preserve location information"
                )
    
    def _apply_user_protections(self, user_protections: Dict[str, str]):
        """Apply user-specified protections."""
        level_map = {
            "critical": ProtectionLevel.CRITICAL,
            "protected": ProtectionLevel.PROTECTED,
            "careful": ProtectionLevel.CAREFUL,
            "flexible": ProtectionLevel.FLEXIBLE,
            "structural": ProtectionLevel.STRUCTURAL,
        }
        
        for col, level_str in user_protections.items():
            if col in self.protections and level_str.lower() in level_map:
                self._set_protection(
                    col,
                    level_map[level_str.lower()],
                    "User-specified protection",
                    upgrade_only=False  # User can override
                )
    
    def _update_allowed_operations(self):
        """Update allowed operations based on final protection levels."""
        for col, protection in self.protections.items():
            protection.allowed_operations = self.ALLOWED_OPERATIONS[protection.level].copy()
            protection.forbidden_operations = self.FORBIDDEN_OPERATIONS.copy()
    
    def is_operation_allowed(self, col: str, operation: str) -> bool:
        """Check if an operation is allowed on a column."""
        if col not in self.protections:
            return True  # Unknown column - allow by default
        
        protection = self.protections[col]
        
        # Check forbidden operations first
        if operation in protection.forbidden_operations:
            return False
        
        # Check allowed operations
        return operation in protection.allowed_operations
    
    def get_protection_level(self, col: str) -> ProtectionLevel:
        """Get protection level for a column."""
        if col in self.protections:
            return self.protections[col].level
        return ProtectionLevel.FLEXIBLE
    
    def get_columns_by_level(self, level: ProtectionLevel) -> List[str]:
        """Get all columns at a specific protection level."""
        return [
            col for col, protection in self.protections.items()
            if protection.level == level
        ]
    
    def get_protection_summary(self) -> Dict[str, Any]:
        """Get summary of all protections."""
        summary = {
            "total_columns": len(self.protections),
            "by_level": {},
            "columns": {},
        }
        
        for level in ProtectionLevel:
            cols = self.get_columns_by_level(level)
            summary["by_level"][level.value] = len(cols)
        
        for col, protection in self.protections.items():
            summary["columns"][col] = {
                "level": protection.level.value,
                "reasons": protection.reasons,
                "allowed_operations": protection.allowed_operations,
            }
        
        return summary
    
    def validate_transformation_plan(
        self,
        transformation_plan: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a transformation plan against protection rules.
        
        Parameters
        ----------
        transformation_plan : list
            List of {column, operation, params} dicts
            
        Returns
        -------
        (is_valid, violations)
        """
        violations = []
        
        for transform in transformation_plan:
            col = transform.get("column")
            operation = transform.get("operation")
            
            if not self.is_operation_allowed(col, operation):
                protection = self.protections.get(col)
                level = protection.level.value if protection else "unknown"
                violations.append(
                    f"Operation '{operation}' not allowed on column '{col}' "
                    f"(protection level: {level})"
                )
        
        return len(violations) == 0, violations
    
    def get_safe_operations(self, col: str) -> List[str]:
        """Get list of safe operations for a column."""
        if col in self.protections:
            return self.protections[col].allowed_operations
        return self.ALLOWED_OPERATIONS[ProtectionLevel.FLEXIBLE]


# Type alias for external use
from typing import Tuple
