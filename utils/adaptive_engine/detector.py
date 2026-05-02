"""
ADIE — Dataset Type and Domain Detection
==========================================

Intelligent detection of dataset type and domain using structural
analysis and semantic pattern matching.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re


class DatasetTypeDetector:
    """
    Classifies datasets into types that determine cleaning strategy:
    - tabular: Generic structured data
    - relational: Entity-based (airports, customers, products)
    - time_series: Temporal dependency present
    - high_cardinality: Dominated by high-cardinality categoricals
    - mixed_type: Complex mixed data types
    """
    
    # Type detection weights
    WEIGHTS = {
        "time_series": {
            "datetime_cols": 0.4,
            "time_keywords": 0.3,
            "sequential_index": 0.2,
            "lag_correlation": 0.1,
        },
        "relational": {
            "identifier_cols": 0.35,
            "foreign_key_pattern": 0.25,
            "entity_keywords": 0.25,
            "normalized_structure": 0.15,
        },
        "high_cardinality": {
            "high_card_ratio": 0.5,
            "text_heavy": 0.3,
            "unique_ratio": 0.2,
        },
    }
    
    def detect_type(self, df: pd.DataFrame, profile: Dict) -> Dict[str, Any]:
        """
        Detect the dataset type based on structural analysis.
        
        Returns
        -------
        {
            "primary_type": str,
            "confidence": float,
            "type_scores": dict,
            "characteristics": dict,
            "recommendations": list
        }
        """
        scores = {
            "tabular": 0.5,  # Base score for generic tabular
            "relational": 0.0,
            "time_series": 0.0,
            "high_cardinality": 0.0,
            "mixed_type": 0.0,
        }
        
        characteristics = {}
        
        # Time-series detection
        ts_score, ts_chars = self._detect_time_series(df, profile)
        scores["time_series"] = ts_score
        characteristics["time_series"] = ts_chars
        
        # Relational detection
        rel_score, rel_chars = self._detect_relational(df, profile)
        scores["relational"] = rel_score
        characteristics["relational"] = rel_chars
        
        # High-cardinality detection
        hc_score, hc_chars = self._detect_high_cardinality(df, profile)
        scores["high_cardinality"] = hc_score
        characteristics["high_cardinality"] = hc_chars
        
        # Mixed-type detection
        mt_score, mt_chars = self._detect_mixed_type(df, profile)
        scores["mixed_type"] = mt_score
        characteristics["mixed_type"] = mt_chars
        
        # Determine primary type
        primary_type = max(scores, key=scores.get)
        confidence = scores[primary_type]
        
        # Generate recommendations based on type
        recommendations = self._generate_recommendations(primary_type, characteristics)
        
        return {
            "primary_type": primary_type,
            "confidence": round(confidence, 3),
            "type_scores": {k: round(v, 3) for k, v in scores.items()},
            "characteristics": characteristics,
            "recommendations": recommendations,
        }
    
    def _detect_time_series(self, df: pd.DataFrame, profile: Dict) -> Tuple[float, Dict]:
        """Detect time-series characteristics."""
        score = 0.0
        chars = {
            "datetime_columns": [],
            "time_keywords_found": [],
            "has_sequential_pattern": False,
            "temporal_granularity": None,
        }
        
        # Check for datetime columns
        datetime_cols = profile.get("datetime_cols", [])
        if datetime_cols:
            score += 0.4 * min(len(datetime_cols) / 2, 1.0)
            chars["datetime_columns"] = datetime_cols
        
        # Check for time-related keywords in column names
        time_keywords = ['date', 'time', 'year', 'month', 'day', 'hour', 'minute',
                        'timestamp', 'period', 'quarter', 'week', 'created', 'updated']
        col_names_lower = [c.lower() for c in df.columns]
        
        found_keywords = []
        for kw in time_keywords:
            for col in col_names_lower:
                if kw in col:
                    found_keywords.append(kw)
                    break
        
        if found_keywords:
            score += 0.3 * min(len(found_keywords) / 3, 1.0)
            chars["time_keywords_found"] = list(set(found_keywords))
        
        # Check for sequential index pattern
        if df.index.is_monotonic_increasing or df.index.is_monotonic_decreasing:
            score += 0.1
            chars["has_sequential_pattern"] = True
        
        # Detect temporal granularity if datetime column exists
        if datetime_cols and len(datetime_cols) > 0:
            try:
                dt_col = datetime_cols[0]
                if dt_col in df.columns:
                    dt_series = pd.to_datetime(df[dt_col], errors='coerce')
                    if dt_series.notna().sum() > 1:
                        diffs = dt_series.dropna().diff().dropna()
                        if len(diffs) > 0:
                            median_diff = diffs.median()
                            if median_diff.days >= 365:
                                chars["temporal_granularity"] = "yearly"
                            elif median_diff.days >= 28:
                                chars["temporal_granularity"] = "monthly"
                            elif median_diff.days >= 7:
                                chars["temporal_granularity"] = "weekly"
                            elif median_diff.days >= 1:
                                chars["temporal_granularity"] = "daily"
                            elif median_diff.seconds >= 3600:
                                chars["temporal_granularity"] = "hourly"
                            else:
                                chars["temporal_granularity"] = "sub_hourly"
            except:
                pass
        
        return score, chars
    
    def _detect_relational(self, df: pd.DataFrame, profile: Dict) -> Tuple[float, Dict]:
        """Detect relational/entity-based characteristics."""
        score = 0.0
        chars = {
            "identifier_columns": [],
            "potential_foreign_keys": [],
            "entity_type": None,
        }
        
        # Check for identifier columns
        cardinality = profile.get("cardinality_analysis", {})
        potential_ids = cardinality.get("potential_identifiers", [])
        
        if potential_ids:
            score += 0.35 * min(len(potential_ids) / 2, 1.0)
            chars["identifier_columns"] = potential_ids
        
        # Check for foreign key patterns (columns ending in _id, _code, etc.)
        fk_patterns = ['_id', '_code', '_key', '_no', '_number']
        col_names_lower = [c.lower() for c in df.columns]
        
        fk_cols = []
        for col in col_names_lower:
            if any(col.endswith(p) for p in fk_patterns):
                fk_cols.append(col)
        
        if fk_cols:
            score += 0.25 * min(len(fk_cols) / 3, 1.0)
            chars["potential_foreign_keys"] = fk_cols
        
        # Detect entity type from column names
        entity_keywords = {
            "customer": ["customer", "client", "user", "member", "subscriber"],
            "product": ["product", "item", "sku", "article", "goods"],
            "transaction": ["transaction", "order", "purchase", "sale", "invoice"],
            "location": ["airport", "station", "store", "branch", "location", "city"],
            "employee": ["employee", "staff", "worker", "personnel"],
        }
        
        col_text = " ".join(col_names_lower)
        for entity_type, keywords in entity_keywords.items():
            if any(kw in col_text for kw in keywords):
                chars["entity_type"] = entity_type
                score += 0.25
                break
        
        return score, chars
    
    def _detect_high_cardinality(self, df: pd.DataFrame, profile: Dict) -> Tuple[float, Dict]:
        """Detect high-cardinality categorical dataset."""
        score = 0.0
        chars = {
            "high_cardinality_columns": [],
            "avg_cardinality": 0,
            "text_heavy_columns": [],
        }
        
        cardinality = profile.get("cardinality_analysis", {})
        high_card_cols = cardinality.get("high_cardinality_columns", [])
        
        # Calculate ratio of high-cardinality columns
        categorical_cols = profile.get("categorical_cols", [])
        if categorical_cols:
            hc_ratio = len(high_card_cols) / len(categorical_cols)
            score += 0.5 * hc_ratio
            chars["high_cardinality_columns"] = high_card_cols
        
        # Check for text-heavy columns
        text_cols = []
        for col in df.select_dtypes(include='object').columns:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 50:
                text_cols.append(col)
        
        if text_cols:
            score += 0.3 * min(len(text_cols) / 3, 1.0)
            chars["text_heavy_columns"] = text_cols
        
        # Calculate average cardinality
        if categorical_cols:
            avg_card = np.mean([df[c].nunique() for c in categorical_cols if c in df.columns])
            chars["avg_cardinality"] = round(avg_card, 1)
            if avg_card > 100:
                score += 0.2
        
        return score, chars
    
    def _detect_mixed_type(self, df: pd.DataFrame, profile: Dict) -> Tuple[float, Dict]:
        """Detect mixed-type dataset characteristics."""
        score = 0.0
        chars = {
            "mixed_columns": [],
            "type_distribution": {},
        }
        
        # Check for columns with mixed content
        mixed_cols = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Check for mixed numeric/text
                numeric_mask = pd.to_numeric(sample, errors='coerce').notna()
                numeric_ratio = numeric_mask.sum() / len(sample)
                if 0.1 < numeric_ratio < 0.9:
                    mixed_cols.append(col)
        
        if mixed_cols:
            score += 0.4 * min(len(mixed_cols) / 3, 1.0)
            chars["mixed_columns"] = mixed_cols
        
        # Calculate type distribution
        type_counts = {
            "numeric": len(profile.get("numeric_cols", [])),
            "categorical": len(profile.get("categorical_cols", [])),
            "datetime": len(profile.get("datetime_cols", [])),
            "boolean": len(profile.get("boolean_cols", [])),
        }
        chars["type_distribution"] = type_counts
        
        # High diversity of types indicates mixed dataset
        non_zero_types = sum(1 for v in type_counts.values() if v > 0)
        if non_zero_types >= 3:
            score += 0.3
        
        return score, chars
    
    def _generate_recommendations(self, primary_type: str, characteristics: Dict) -> List[str]:
        """Generate cleaning recommendations based on dataset type."""
        recommendations = []
        
        if primary_type == "time_series":
            recommendations.extend([
                "Preserve chronological order during train/test split",
                "Consider lag features for temporal patterns",
                "Avoid random shuffling to prevent data leakage",
                "Use time-aware cross-validation",
            ])
            if characteristics.get("time_series", {}).get("temporal_granularity"):
                gran = characteristics["time_series"]["temporal_granularity"]
                recommendations.append(f"Detected {gran} granularity - consider appropriate aggregation")
        
        elif primary_type == "relational":
            recommendations.extend([
                "Preserve entity identifiers for traceability",
                "Consider entity-based feature engineering",
                "Maintain referential integrity during cleaning",
            ])
            if characteristics.get("relational", {}).get("entity_type"):
                entity = characteristics["relational"]["entity_type"]
                recommendations.append(f"Detected {entity} entity - apply domain-specific rules")
        
        elif primary_type == "high_cardinality":
            recommendations.extend([
                "Use frequency or target encoding for high-cardinality columns",
                "Consider hash encoding for very high cardinality",
                "Avoid one-hot encoding to prevent dimensionality explosion",
            ])
        
        elif primary_type == "mixed_type":
            recommendations.extend([
                "Apply column-specific type coercion",
                "Handle mixed columns with careful parsing",
                "Consider creating separate features for different types",
            ])
        
        else:  # tabular
            recommendations.extend([
                "Apply standard preprocessing pipeline",
                "Use appropriate encoding based on cardinality",
                "Consider feature interactions",
            ])
        
        return recommendations


class DomainDetector:
    """
    Detects the domain/industry of a dataset using semantic analysis
    of column names and value patterns.
    """
    
    # Domain keyword mappings with weights
    DOMAIN_KEYWORDS = {
        "aviation": {
            "keywords": [
                "flight", "airport", "icao", "iata", "airline", "aircraft",
                "departure", "arrival", "runway", "terminal", "gate",
                "passenger", "cargo", "pilot", "crew", "altitude",
                "apt", "flt", "dep", "arr", "atc", "airspace"
            ],
            "weight": 1.0,
        },
        "finance": {
            "keywords": [
                "revenue", "sales", "price", "cost", "profit", "margin",
                "transaction", "payment", "invoice", "balance", "credit",
                "debit", "loan", "interest", "rate", "currency", "amount",
                "fiscal", "budget", "expense", "income", "tax", "fee"
            ],
            "weight": 1.0,
        },
        "healthcare": {
            "keywords": [
                "patient", "diagnosis", "treatment", "medication", "drug",
                "symptom", "disease", "hospital", "clinic", "doctor",
                "nurse", "medical", "health", "blood", "pressure", "pulse",
                "bmi", "cholesterol", "glucose", "prescription", "dosage"
            ],
            "weight": 1.0,
        },
        "marketing": {
            "keywords": [
                "campaign", "customer", "conversion", "click", "impression",
                "lead", "funnel", "segment", "channel", "engagement",
                "retention", "churn", "acquisition", "lifetime", "value",
                "email", "social", "ad", "promotion", "discount"
            ],
            "weight": 1.0,
        },
        "ecommerce": {
            "keywords": [
                "product", "order", "cart", "checkout", "shipping",
                "inventory", "sku", "category", "brand", "review",
                "rating", "wishlist", "return", "refund", "supplier"
            ],
            "weight": 1.0,
        },
        "hr": {
            "keywords": [
                "employee", "salary", "department", "position", "hire",
                "termination", "performance", "review", "attendance",
                "leave", "vacation", "benefits", "training", "promotion"
            ],
            "weight": 1.0,
        },
        "manufacturing": {
            "keywords": [
                "machine", "production", "defect", "quality", "batch",
                "assembly", "downtime", "maintenance", "sensor", "output",
                "yield", "scrap", "efficiency", "cycle", "shift"
            ],
            "weight": 1.0,
        },
        "logistics": {
            "keywords": [
                "shipment", "delivery", "warehouse", "route", "tracking",
                "carrier", "freight", "container", "dispatch", "fleet",
                "driver", "destination", "origin", "eta", "weight"
            ],
            "weight": 1.0,
        },
        "education": {
            "keywords": [
                "student", "course", "grade", "enrollment", "teacher",
                "class", "semester", "exam", "score", "attendance",
                "curriculum", "degree", "major", "gpa", "credit"
            ],
            "weight": 1.0,
        },
        "real_estate": {
            "keywords": [
                "property", "listing", "price", "sqft", "bedroom",
                "bathroom", "location", "neighborhood", "rent", "mortgage",
                "agent", "buyer", "seller", "closing", "appraisal"
            ],
            "weight": 1.0,
        },
    }
    
    # Domain-specific protected columns
    DOMAIN_PROTECTED_COLUMNS = {
        "aviation": [
            "icao", "iata", "airport", "flight", "airline",
            "departure", "arrival", "apt", "flt"
        ],
        "finance": [
            "transaction_id", "account", "amount", "currency",
            "date", "balance"
        ],
        "healthcare": [
            "patient_id", "diagnosis", "treatment", "medication",
            "date", "doctor"
        ],
        "marketing": [
            "customer_id", "campaign_id", "channel", "date",
            "conversion"
        ],
    }
    
    def detect_domain(self, df: pd.DataFrame, profile: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect the domain of the dataset.
        
        Returns
        -------
        {
            "domain": str,
            "confidence": float,
            "domain_scores": dict,
            "matched_keywords": dict,
            "protected_columns": list,
            "feature_engineering_suggestions": list
        }
        """
        col_names = df.columns.tolist()
        col_names_lower = [c.lower() for c in col_names]
        col_text = " ".join(col_names_lower)
        
        # Score each domain
        domain_scores = {}
        matched_keywords = {}
        
        for domain, config in self.DOMAIN_KEYWORDS.items():
            keywords = config["keywords"]
            weight = config["weight"]
            
            matches = []
            for kw in keywords:
                # Check exact column match
                if kw in col_names_lower:
                    matches.append((kw, "exact_column"))
                # Check partial match in column names
                elif any(kw in col for col in col_names_lower):
                    matches.append((kw, "partial_column"))
                # Check in column text
                elif kw in col_text:
                    matches.append((kw, "text_match"))
            
            if matches:
                # Calculate score based on match quality
                exact_matches = sum(1 for _, t in matches if t == "exact_column")
                partial_matches = sum(1 for _, t in matches if t == "partial_column")
                text_matches = sum(1 for _, t in matches if t == "text_match")
                
                score = (exact_matches * 1.0 + partial_matches * 0.7 + text_matches * 0.3) / len(keywords)
                score = min(score * weight * 2, 1.0)  # Scale and cap at 1.0
                
                domain_scores[domain] = round(score, 3)
                matched_keywords[domain] = [kw for kw, _ in matches]
        
        # Determine primary domain
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[primary_domain]
        else:
            primary_domain = "general"
            confidence = 0.5
        
        # Get protected columns for this domain
        protected_cols = self._get_protected_columns(primary_domain, col_names_lower)
        
        # Generate feature engineering suggestions
        fe_suggestions = self._get_feature_engineering_suggestions(primary_domain, df)
        
        return {
            "domain": primary_domain,
            "confidence": confidence,
            "domain_scores": domain_scores,
            "matched_keywords": matched_keywords,
            "protected_columns": protected_cols,
            "feature_engineering_suggestions": fe_suggestions,
        }
    
    def _get_protected_columns(self, domain: str, col_names_lower: List[str]) -> List[str]:
        """Get columns that should be protected based on domain."""
        protected = []
        
        domain_protected = self.DOMAIN_PROTECTED_COLUMNS.get(domain, [])
        
        for col in col_names_lower:
            for protected_kw in domain_protected:
                if protected_kw in col:
                    protected.append(col)
                    break
        
        return list(set(protected))
    
    def _get_feature_engineering_suggestions(self, domain: str, df: pd.DataFrame) -> List[Dict]:
        """Generate domain-specific feature engineering suggestions."""
        suggestions = []
        col_names_lower = [c.lower() for c in df.columns]
        
        if domain == "aviation":
            # Traffic volume features
            dep_cols = [c for c in col_names_lower if 'dep' in c and any(x in c for x in ['count', 'num', 'total'])]
            arr_cols = [c for c in col_names_lower if 'arr' in c and any(x in c for x in ['count', 'num', 'total'])]
            
            if dep_cols and arr_cols:
                suggestions.append({
                    "name": "traffic_volume",
                    "formula": f"{dep_cols[0]} + {arr_cols[0]}",
                    "description": "Total traffic volume (departures + arrivals)",
                })
                suggestions.append({
                    "name": "dep_arr_ratio",
                    "formula": f"{dep_cols[0]} / ({arr_cols[0]} + 1)",
                    "description": "Departure to arrival ratio",
                })
        
        elif domain == "finance":
            # Profit margin
            if any('revenue' in c for c in col_names_lower) and any('cost' in c for c in col_names_lower):
                suggestions.append({
                    "name": "profit_margin",
                    "formula": "(revenue - cost) / revenue",
                    "description": "Profit margin percentage",
                })
        
        elif domain == "marketing":
            # Conversion rate
            if any('click' in c for c in col_names_lower) and any('conversion' in c for c in col_names_lower):
                suggestions.append({
                    "name": "conversion_rate",
                    "formula": "conversions / (clicks + 1)",
                    "description": "Click to conversion rate",
                })
        
        elif domain == "ecommerce":
            # Average order value
            if any('order' in c for c in col_names_lower) and any('amount' in c or 'total' in c for c in col_names_lower):
                suggestions.append({
                    "name": "avg_order_value",
                    "formula": "total_amount / n_orders",
                    "description": "Average order value",
                })
        
        return suggestions
