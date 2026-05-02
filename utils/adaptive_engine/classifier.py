"""
ADIE — Column Role Classifier
==============================

Intelligent classification of each column's role in the dataset,
overriding simple dtype-based logic with semantic understanding.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re


class ColumnRoleClassifier:
    """
    Classifies each column into a semantic role:
    - target: The prediction target
    - identifier: Unique identifiers (IDs, codes)
    - temporal: Date/time columns
    - geographical: Location-based columns
    - categorical_nominal: Unordered categories
    - categorical_ordinal: Ordered categories
    - numerical_continuous: Continuous numeric values
    - numerical_discrete: Discrete numeric values (counts, etc.)
    - derived: Computed/derived columns
    - text: Free-text columns
    - binary: Binary flags/indicators
    """
    
    # Keywords for role detection
    ROLE_KEYWORDS = {
        "identifier": [
            "id", "_id", "code", "key", "number", "no", "num",
            "uuid", "guid", "hash", "token", "index", "serial"
        ],
        "temporal": [
            "date", "time", "timestamp", "datetime", "year", "month",
            "day", "hour", "minute", "second", "week", "quarter",
            "created", "updated", "modified", "start", "end", "period"
        ],
        "geographical": [
            "lat", "latitude", "lon", "longitude", "city", "state",
            "country", "region", "zip", "postal", "address", "location",
            "geo", "place", "area", "district", "province", "county"
        ],
        "ordinal": [
            "level", "grade", "rank", "tier", "priority", "severity",
            "rating", "score", "stage", "phase", "class", "quality"
        ],
        "binary": [
            "is_", "has_", "flag", "indicator", "active", "enabled",
            "valid", "approved", "confirmed", "success", "failed"
        ],
        "text": [
            "description", "comment", "note", "remark", "text", "body",
            "content", "message", "summary", "detail", "narrative"
        ],
        "derived": [
            "total", "sum", "avg", "average", "mean", "count", "ratio",
            "percentage", "pct", "rate", "diff", "delta", "change"
        ],
    }
    
    # Ordinal value patterns
    ORDINAL_PATTERNS = [
        ["low", "medium", "high"],
        ["small", "medium", "large"],
        ["poor", "fair", "good", "excellent"],
        ["bad", "average", "good", "great"],
        ["junior", "mid", "senior", "lead", "principal"],
        ["bronze", "silver", "gold", "platinum"],
        ["basic", "standard", "premium", "enterprise"],
        ["1", "2", "3", "4", "5"],
        ["a", "b", "c", "d", "e", "f"],
    ]
    
    def classify_columns(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        profile: Optional[Dict] = None,
        domain_info: Optional[Dict] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify all columns in the dataset.
        
        Returns
        -------
        {
            "column_name": {
                "role": str,
                "confidence": float,
                "sub_type": str | None,
                "reasons": list,
                "recommendations": list
            },
            ...
        }
        """
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        if target_col:
            target_col = target_col.strip()
        
        classifications = {}
        
        for col in df.columns:
            classification = self._classify_single_column(
                df, col, target_col, profile, domain_info
            )
            classifications[col] = classification
        
        # Post-process to resolve conflicts
        classifications = self._resolve_conflicts(classifications, df)
        
        return classifications
    
    def _classify_single_column(
        self,
        df: pd.DataFrame,
        col: str,
        target_col: Optional[str],
        profile: Optional[Dict],
        domain_info: Optional[Dict]
    ) -> Dict[str, Any]:
        """Classify a single column."""
        series = df[col]
        col_lower = col.lower()
        
        # Initialize scores for each role
        role_scores = {
            "target": 0.0,
            "identifier": 0.0,
            "temporal": 0.0,
            "geographical": 0.0,
            "categorical_nominal": 0.0,
            "categorical_ordinal": 0.0,
            "numerical_continuous": 0.0,
            "numerical_discrete": 0.0,
            "derived": 0.0,
            "text": 0.0,
            "binary": 0.0,
        }
        
        reasons = []
        
        # Check if this is the target column
        if col == target_col:
            role_scores["target"] = 1.0
            reasons.append("Explicitly specified as target column")
        
        # Keyword-based scoring
        keyword_scores, keyword_reasons = self._score_by_keywords(col_lower)
        for role, score in keyword_scores.items():
            role_scores[role] += score
        reasons.extend(keyword_reasons)
        
        # Data type-based scoring
        dtype_scores, dtype_reasons = self._score_by_dtype(series)
        for role, score in dtype_scores.items():
            role_scores[role] += score
        reasons.extend(dtype_reasons)
        
        # Cardinality-based scoring
        card_scores, card_reasons = self._score_by_cardinality(series, len(df))
        for role, score in card_scores.items():
            role_scores[role] += score
        reasons.extend(card_reasons)
        
        # Value pattern-based scoring
        pattern_scores, pattern_reasons = self._score_by_patterns(series)
        for role, score in pattern_scores.items():
            role_scores[role] += score
        reasons.extend(pattern_reasons)
        
        # Domain-specific adjustments
        if domain_info:
            domain_scores, domain_reasons = self._score_by_domain(
                col_lower, domain_info
            )
            for role, score in domain_scores.items():
                role_scores[role] += score
            reasons.extend(domain_reasons)
        
        # Determine primary role
        primary_role = max(role_scores, key=role_scores.get)
        confidence = min(role_scores[primary_role], 1.0)
        
        # Determine sub-type
        sub_type = self._determine_subtype(series, primary_role)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(primary_role, series, col)
        
        return {
            "role": primary_role,
            "confidence": round(confidence, 3),
            "sub_type": sub_type,
            "reasons": reasons,
            "recommendations": recommendations,
            "role_scores": {k: round(v, 3) for k, v in role_scores.items()},
        }
    
    def _score_by_keywords(self, col_lower: str) -> Tuple[Dict[str, float], List[str]]:
        """Score based on column name keywords."""
        scores = {}
        reasons = []
        
        for role, keywords in self.ROLE_KEYWORDS.items():
            for kw in keywords:
                if col_lower == kw or col_lower.startswith(kw) or col_lower.endswith(kw):
                    scores[role] = scores.get(role, 0) + 0.4
                    reasons.append(f"Column name matches '{kw}' keyword for {role}")
                    break
                elif kw in col_lower:
                    scores[role] = scores.get(role, 0) + 0.2
                    reasons.append(f"Column name contains '{kw}' suggesting {role}")
                    break
        
        return scores, reasons
    
    def _score_by_dtype(self, series: pd.Series) -> Tuple[Dict[str, float], List[str]]:
        """Score based on data type."""
        scores = {}
        reasons = []
        
        if pd.api.types.is_datetime64_any_dtype(series):
            scores["temporal"] = 0.5
            reasons.append("Column has datetime dtype")
        
        elif pd.api.types.is_bool_dtype(series):
            scores["binary"] = 0.5
            reasons.append("Column has boolean dtype")
        
        elif pd.api.types.is_integer_dtype(series):
            scores["numerical_discrete"] = 0.3
            reasons.append("Column has integer dtype")
        
        elif pd.api.types.is_float_dtype(series):
            scores["numerical_continuous"] = 0.3
            reasons.append("Column has float dtype")
        
        elif series.dtype == 'object':
            scores["categorical_nominal"] = 0.2
            reasons.append("Column has object dtype (likely categorical)")
        
        return scores, reasons
    
    def _score_by_cardinality(self, series: pd.Series, n_rows: int) -> Tuple[Dict[str, float], List[str]]:
        """Score based on cardinality patterns."""
        scores = {}
        reasons = []
        
        n_unique = series.nunique()
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0
        
        if n_unique == 2:
            scores["binary"] = 0.4
            reasons.append("Binary cardinality (2 unique values)")
        
        elif unique_ratio > 0.95 and n_unique > 100:
            scores["identifier"] = 0.4
            reasons.append(f"Very high cardinality ({unique_ratio:.1%} unique) suggests identifier")
        
        elif n_unique <= 10:
            scores["categorical_nominal"] = 0.2
            scores["categorical_ordinal"] = 0.1
            reasons.append(f"Low cardinality ({n_unique} unique) suggests categorical")
        
        elif n_unique <= 50:
            scores["categorical_nominal"] = 0.15
            reasons.append(f"Medium cardinality ({n_unique} unique)")
        
        return scores, reasons
    
    def _score_by_patterns(self, series: pd.Series) -> Tuple[Dict[str, float], List[str]]:
        """Score based on value patterns."""
        scores = {}
        reasons = []
        
        if series.dtype == 'object':
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return scores, reasons
            
            sample_lower = sample.astype(str).str.lower()
            unique_lower = sample_lower.unique()
            
            # Check for ordinal patterns
            for pattern in self.ORDINAL_PATTERNS:
                if any(p in unique_lower for p in pattern):
                    scores["categorical_ordinal"] = 0.3
                    reasons.append(f"Values match ordinal pattern: {pattern[:3]}...")
                    break
            
            # Check for date patterns
            date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}'
            if sample.astype(str).str.match(date_pattern).mean() > 0.7:
                scores["temporal"] = 0.4
                reasons.append("Values match date pattern")
            
            # Check for ID patterns (alphanumeric codes)
            id_pattern = r'^[A-Z]{2,4}[-_]?\d{3,}$|^\d{5,}$|^[A-Z0-9]{8,}$'
            if sample.astype(str).str.match(id_pattern, case=False).mean() > 0.5:
                scores["identifier"] = 0.3
                reasons.append("Values match identifier pattern")
            
            # Check for text (long strings)
            avg_len = sample.astype(str).str.len().mean()
            if avg_len > 100:
                scores["text"] = 0.4
                reasons.append(f"Long average string length ({avg_len:.0f}) suggests text")
        
        elif pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                # Check if values are integer-like
                if np.allclose(non_null, non_null.round()):
                    # Check if it looks like counts
                    if (non_null >= 0).all() and non_null.max() < 1000:
                        scores["numerical_discrete"] = 0.2
                        reasons.append("Non-negative integers suggest discrete counts")
                else:
                    scores["numerical_continuous"] = 0.2
                    reasons.append("Non-integer values suggest continuous")
                
                # Check for latitude/longitude ranges
                if -90 <= non_null.min() and non_null.max() <= 90:
                    scores["geographical"] = 0.1
                    reasons.append("Value range suggests latitude")
                elif -180 <= non_null.min() and non_null.max() <= 180:
                    scores["geographical"] = 0.1
                    reasons.append("Value range suggests longitude")
        
        return scores, reasons
    
    def _score_by_domain(self, col_lower: str, domain_info: Dict) -> Tuple[Dict[str, float], List[str]]:
        """Score based on domain context."""
        scores = {}
        reasons = []
        
        domain = domain_info.get("domain", "general")
        protected_cols = domain_info.get("protected_columns", [])
        
        # Boost identifier score for protected columns
        if col_lower in protected_cols:
            scores["identifier"] = scores.get("identifier", 0) + 0.2
            reasons.append(f"Column is domain-protected in {domain}")
        
        return scores, reasons
    
    def _determine_subtype(self, series: pd.Series, primary_role: str) -> Optional[str]:
        """Determine sub-type within the primary role."""
        if primary_role == "numerical_continuous":
            non_null = series.dropna()
            if len(non_null) > 0:
                if (non_null >= 0).all() and (non_null <= 1).all():
                    return "proportion"
                elif (non_null >= 0).all() and (non_null <= 100).all():
                    return "percentage"
                elif (non_null > 0).all():
                    return "positive_continuous"
            return "general_continuous"
        
        elif primary_role == "numerical_discrete":
            non_null = series.dropna()
            if len(non_null) > 0:
                if (non_null >= 0).all():
                    return "count"
                elif series.nunique() <= 10:
                    return "ordinal_numeric"
            return "general_discrete"
        
        elif primary_role == "categorical_nominal":
            n_unique = series.nunique()
            if n_unique <= 5:
                return "low_cardinality"
            elif n_unique <= 20:
                return "medium_cardinality"
            elif n_unique <= 100:
                return "high_cardinality"
            else:
                return "very_high_cardinality"
        
        elif primary_role == "temporal":
            if pd.api.types.is_datetime64_any_dtype(series):
                return "datetime"
            return "date_string"
        
        elif primary_role == "identifier":
            sample = series.dropna().head(10).astype(str)
            if sample.str.isnumeric().all():
                return "numeric_id"
            elif sample.str.match(r'^[A-Z]+$').all():
                return "code"
            else:
                return "alphanumeric_id"
        
        return None
    
    def _generate_recommendations(self, role: str, series: pd.Series, col: str) -> List[str]:
        """Generate processing recommendations for the column."""
        recommendations = []
        
        if role == "identifier":
            recommendations.extend([
                "Preserve for traceability - DO NOT DROP",
                "Apply label encoding or hash encoding",
                "Consider keeping original for debugging",
            ])
        
        elif role == "temporal":
            recommendations.extend([
                "Extract temporal features (year, month, day, weekday)",
                "Preserve original for time-series analysis",
                "Consider cyclical encoding for periodic features",
            ])
        
        elif role == "geographical":
            recommendations.extend([
                "Preserve location information",
                "Consider geospatial feature engineering",
                "May benefit from clustering or binning",
            ])
        
        elif role == "categorical_nominal":
            n_unique = series.nunique()
            if n_unique <= 10:
                recommendations.append("Use one-hot encoding")
            elif n_unique <= 50:
                recommendations.append("Use target encoding or ordinal encoding")
            else:
                recommendations.append("Use frequency encoding or hash encoding")
        
        elif role == "categorical_ordinal":
            recommendations.extend([
                "Use ordinal encoding preserving order",
                "Map to numeric scale based on domain knowledge",
            ])
        
        elif role == "numerical_continuous":
            skewness = series.dropna().skew() if len(series.dropna()) > 2 else 0
            if abs(skewness) > 2:
                recommendations.append("Consider log or Box-Cox transformation")
            recommendations.append("Apply scaling (StandardScaler or RobustScaler)")
        
        elif role == "numerical_discrete":
            recommendations.extend([
                "May benefit from binning",
                "Consider treating as categorical if few unique values",
            ])
        
        elif role == "text":
            recommendations.extend([
                "Consider text vectorization (TF-IDF, embeddings)",
                "Extract text features (length, word count)",
                "May need NLP preprocessing",
            ])
        
        elif role == "binary":
            recommendations.extend([
                "Keep as 0/1 encoding",
                "No transformation needed",
            ])
        
        return recommendations
    
    def _resolve_conflicts(self, classifications: Dict, df: pd.DataFrame) -> Dict:
        """Resolve classification conflicts and ensure consistency."""
        # Ensure only one target
        targets = [col for col, info in classifications.items() if info["role"] == "target"]
        if len(targets) > 1:
            # Keep the one with highest confidence
            best_target = max(targets, key=lambda c: classifications[c]["confidence"])
            for col in targets:
                if col != best_target:
                    # Reclassify as the second-best role
                    scores = classifications[col]["role_scores"]
                    scores["target"] = 0
                    new_role = max(scores, key=scores.get)
                    classifications[col]["role"] = new_role
                    classifications[col]["reasons"].append("Reclassified: only one target allowed")
        
        return classifications
    
    def get_columns_by_role(self, classifications: Dict) -> Dict[str, List[str]]:
        """Group columns by their classified role."""
        by_role = {}
        for col, info in classifications.items():
            role = info["role"]
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(col)
        return by_role
