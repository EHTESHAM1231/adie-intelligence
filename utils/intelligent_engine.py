"""
ADIE — Intelligent Decision Engine
Replaces static rule-based logic with a weighted, data-driven strategy selector.

Every public function returns a structured dict that includes:
  - decision  : what was chosen
  - confidence: float in [0, 1]
  - reason    : human-readable explanation
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET PROFILER
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target_col: str) -> dict:
    """
    Produce a rich, structured profile of the dataset.
    Used downstream by the strategy selector.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    target_col = target_col.strip()

    n_rows, n_cols = df.shape
    feature_cols = [c for c in df.columns if c != target_col]

    # ── Missing values ────────────────────────────────────────────────────
    missing_per_col = df.isnull().sum()
    missing_ratio_per_col = (missing_per_col / n_rows).to_dict()
    total_missing = int(missing_per_col.sum())
    overall_missing_ratio = total_missing / (n_rows * n_cols) if n_rows * n_cols > 0 else 0.0

    # ── Data type distribution ────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include='bool').columns.tolist()

    dtype_distribution = {
        "numeric": len(numeric_cols),
        "categorical": len(categorical_cols),
        "boolean": len(bool_cols),
        "other": n_cols - len(numeric_cols) - len(categorical_cols) - len(bool_cols)
    }

    # ── Cardinality of categorical columns ───────────────────────────────
    cardinality = {}
    for col in categorical_cols:
        if col in feature_cols:
            cardinality[col] = int(df[col].nunique())

    high_cardinality_cols = [c for c, v in cardinality.items() if v > 50]
    max_cardinality = max(cardinality.values()) if cardinality else 0

    # ── Skewness of numeric features ─────────────────────────────────────
    skewness = {}
    for col in numeric_cols:
        if col in feature_cols:
            try:
                sk = float(df[col].dropna().skew())
                skewness[col] = round(sk, 4)
            except Exception:
                skewness[col] = 0.0

    highly_skewed = [c for c, v in skewness.items() if abs(v) > 1.0]
    avg_skewness = float(np.mean(list(skewness.values()))) if skewness else 0.0

    # ── Correlation matrix density ────────────────────────────────────────
    corr_density = 0.0
    high_corr_pairs = []
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            n_pairs = upper.count().sum()
            n_high = (upper > 0.8).sum().sum()
            corr_density = float(n_high / n_pairs) if n_pairs > 0 else 0.0
            # Collect top correlated pairs
            stacked = upper.stack()
            top_pairs = stacked[stacked > 0.8].sort_values(ascending=False).head(5)
            high_corr_pairs = [
                {"col_a": a, "col_b": b, "correlation": round(float(v), 4)}
                for (a, b), v in top_pairs.items()
            ]
        except Exception:
            pass

    # ── Class balance (target) ────────────────────────────────────────────
    target_info = {}
    if target_col in df.columns:
        n_unique_target = int(df[target_col].nunique())
        value_counts = df[target_col].value_counts(normalize=True)
        imbalance_ratio = float(value_counts.iloc[0] / value_counts.iloc[-1]) if len(value_counts) > 1 else 1.0
        target_info = {
            "unique_values": n_unique_target,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "dominant_class_share": round(float(value_counts.iloc[0]), 4)
        }

    # ── Duplicate rows ────────────────────────────────────────────────────
    n_duplicates = int(df.duplicated().sum())
    duplicate_ratio = n_duplicates / n_rows if n_rows > 0 else 0.0

    # ── Dataset size classification ───────────────────────────────────────
    if n_rows < 500:
        size_class = "tiny"
    elif n_rows < 5_000:
        size_class = "small"
    elif n_rows < 50_000:
        size_class = "medium"
    elif n_rows < 500_000:
        size_class = "large"
    else:
        size_class = "very_large"

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "size_class": size_class,
        "overall_missing_ratio": round(overall_missing_ratio, 4),
        "missing_ratio_per_col": {k: round(v, 4) for k, v in missing_ratio_per_col.items()},
        "total_missing": total_missing,
        "dtype_distribution": dtype_distribution,
        "cardinality": cardinality,
        "high_cardinality_cols": high_cardinality_cols,
        "max_cardinality": max_cardinality,
        "skewness": skewness,
        "highly_skewed_cols": highly_skewed,
        "avg_skewness": round(avg_skewness, 4),
        "corr_density": round(corr_density, 4),
        "high_corr_pairs": high_corr_pairs,
        "target_info": target_info,
        "n_duplicates": n_duplicates,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTELLIGENT STRATEGY SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def select_strategies(profile: dict) -> dict:
    """
    Given a dataset profile, return a full strategy plan with decisions,
    confidence scores, and human-readable reasons.

    Returns
    -------
    {
        "imputation":        {decision, confidence, reason},
        "outlier_handling":  {decision, confidence, reason},
        "encoding":          {decision, confidence, reason},
        "scaling":           {decision, confidence, reason},
        "imbalance":         {decision, confidence, reason},
        "model_selection":   {decision, confidence, reason, recommended_models},
        "diagnostics_focus": {decision, confidence, reason, checks}
    }
    """
    strategies = {}

    # ── Imputation strategy ───────────────────────────────────────────────
    strategies["imputation"] = _decide_imputation(profile)

    # ── Outlier handling ──────────────────────────────────────────────────
    strategies["outlier_handling"] = _decide_outlier_handling(profile)

    # ── Encoding strategy ─────────────────────────────────────────────────
    strategies["encoding"] = _decide_encoding(profile)

    # ── Feature scaling ───────────────────────────────────────────────────
    strategies["scaling"] = _decide_scaling(profile)

    # ── Class imbalance ───────────────────────────────────────────────────
    strategies["imbalance"] = _decide_imbalance(profile)

    # ── Model selection ───────────────────────────────────────────────────
    strategies["model_selection"] = _decide_models(profile)

    # ── Diagnostics focus ─────────────────────────────────────────────────
    strategies["diagnostics_focus"] = _decide_diagnostics(profile)

    return strategies


# ── Individual decision functions ─────────────────────────────────────────────

def _decide_imputation(profile: dict) -> dict:
    missing_ratio = profile["overall_missing_ratio"]
    avg_skew = abs(profile["avg_skewness"])
    size_class = profile["size_class"]

    score_median = 0.0
    score_mean = 0.0
    score_knn = 0.0
    score_iterative = 0.0
    reasons = []

    # Missing ratio drives complexity
    if missing_ratio < 0.02:
        score_median += 0.5
        reasons.append(f"Very low missing rate ({missing_ratio:.1%}) — simple imputation sufficient")
    elif missing_ratio < 0.10:
        score_median += 0.3
        score_knn += 0.2
        reasons.append(f"Moderate missing rate ({missing_ratio:.1%}) — median or KNN imputation")
    elif missing_ratio < 0.30:
        score_knn += 0.4
        score_iterative += 0.3
        reasons.append(f"High missing rate ({missing_ratio:.1%}) — advanced imputation recommended")
    else:
        score_iterative += 0.5
        reasons.append(f"Very high missing rate ({missing_ratio:.1%}) — iterative imputation needed")

    # Skewness favours median over mean
    if avg_skew > 1.0:
        score_median += 0.3
        score_mean -= 0.2
        reasons.append(f"High average skewness ({avg_skew:.2f}) — median preferred over mean")
    else:
        score_mean += 0.2
        reasons.append(f"Low skewness ({avg_skew:.2f}) — mean imputation viable")

    # Dataset size limits KNN
    if size_class in ("tiny", "small"):
        score_knn += 0.1
        reasons.append("Small dataset — KNN imputation feasible")
    elif size_class in ("large", "very_large"):
        score_knn -= 0.3
        reasons.append("Large dataset — KNN imputation too slow; prefer median")

    scores = {
        "Median Imputation": score_median,
        "Mean Imputation": score_mean,
        "KNN Imputation": score_knn,
        "Iterative Imputation": score_iterative
    }
    best = max(scores, key=scores.get)
    confidence = min(1.0, max(0.0, scores[best]))

    return {
        "decision": best,
        "confidence": round(confidence, 3),
        "reason": "; ".join(reasons)
    }


def _decide_outlier_handling(profile: dict) -> dict:
    avg_skew = abs(profile["avg_skewness"])
    n_rows = profile["n_rows"]
    reasons = []

    score_iqr = 0.5       # default — robust and widely understood
    score_zscore = 0.0
    score_winsorize = 0.0
    score_none = 0.0

    if avg_skew > 2.0:
        score_iqr += 0.3
        reasons.append(f"High skewness ({avg_skew:.2f}) — IQR capping handles non-normal distributions well")
    elif avg_skew > 0.5:
        score_iqr += 0.1
        score_winsorize += 0.2
        reasons.append(f"Moderate skewness ({avg_skew:.2f}) — IQR or Winsorization both suitable")
    else:
        score_zscore += 0.3
        reasons.append(f"Low skewness ({avg_skew:.2f}) — Z-score method viable for near-normal data")

    if n_rows < 200:
        score_none += 0.2
        reasons.append("Very small dataset — aggressive outlier removal risks data loss")

    scores = {
        "IQR Capping": score_iqr,
        "Z-Score Removal": score_zscore,
        "Winsorization": score_winsorize,
        "No Outlier Handling": score_none
    }
    best = max(scores, key=scores.get)
    confidence = min(1.0, max(0.0, scores[best]))

    return {
        "decision": best,
        "confidence": round(confidence, 3),
        "reason": "; ".join(reasons)
    }


def _decide_encoding(profile: dict) -> dict:
    max_card = profile["max_cardinality"]
    high_card_cols = profile["high_cardinality_cols"]
    n_cat = profile["dtype_distribution"]["categorical"]
    reasons = []

    if n_cat == 0:
        return {
            "decision": "No Encoding Required",
            "confidence": 1.0,
            "reason": "Dataset has no categorical columns"
        }

    if max_card > 50:
        decision = "Frequency Encoding (high-cardinality) + One-Hot (low-cardinality)"
        confidence = 0.90
        reasons.append(
            f"{len(high_card_cols)} column(s) have >{50} unique values — "
            "frequency encoding prevents one-hot explosion"
        )
    elif max_card > 10:
        decision = "One-Hot Encoding"
        confidence = 0.85
        reasons.append(f"Max cardinality {max_card} — one-hot encoding is safe and interpretable")
    else:
        decision = "One-Hot Encoding"
        confidence = 0.95
        reasons.append(f"Low cardinality ({max_card} max) — one-hot encoding ideal")

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "reason": "; ".join(reasons)
    }


def _decide_scaling(profile: dict) -> dict:
    avg_skew = abs(profile["avg_skewness"])
    n_numeric = profile["dtype_distribution"]["numeric"]
    reasons = []

    if n_numeric == 0:
        return {
            "decision": "No Scaling Required",
            "confidence": 1.0,
            "reason": "No numeric features present"
        }

    if avg_skew > 1.5:
        decision = "RobustScaler"
        confidence = 0.85
        reasons.append(
            f"High skewness ({avg_skew:.2f}) — RobustScaler is resistant to outliers"
        )
    else:
        decision = "StandardScaler"
        confidence = 0.88
        reasons.append(
            f"Moderate skewness ({avg_skew:.2f}) — StandardScaler (zero mean, unit variance) appropriate"
        )

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "reason": "; ".join(reasons)
    }


def _decide_imbalance(profile: dict) -> dict:
    target_info = profile.get("target_info", {})
    imbalance_ratio = target_info.get("imbalance_ratio", 1.0)
    n_rows = profile["n_rows"]
    size_class = profile["size_class"]
    reasons = []

    if imbalance_ratio < 1.5:
        decision = "No Resampling Needed"
        confidence = 0.92
        reasons.append(f"Classes are well-balanced (ratio {imbalance_ratio:.1f}:1)")
    elif imbalance_ratio < 5:
        decision = "SMOTE Oversampling"
        confidence = 0.80
        reasons.append(f"Mild imbalance (ratio {imbalance_ratio:.1f}:1) — SMOTE creates synthetic minority samples")
    elif imbalance_ratio < 20:
        decision = "SMOTE + Tomek Links"
        confidence = 0.82
        reasons.append(f"Moderate imbalance (ratio {imbalance_ratio:.1f}:1) — combined over/under-sampling")
    else:
        decision = "SMOTE Oversampling"
        confidence = 0.75
        reasons.append(
            f"Severe imbalance (ratio {imbalance_ratio:.1f}:1) — SMOTE essential; "
            "consider class-weight adjustment too"
        )

    if size_class == "tiny" and decision != "No Resampling Needed":
        reasons.append("Small dataset — SMOTE may create noisy synthetic samples; monitor carefully")

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "reason": "; ".join(reasons)
    }


def _decide_models(profile: dict) -> dict:
    n_rows = profile["n_rows"]
    n_cols = profile["n_cols"]
    size_class = profile["size_class"]
    target_info = profile.get("target_info", {})
    n_unique_target = target_info.get("unique_values", 2)
    corr_density = profile["corr_density"]
    reasons = []
    recommended = []

    # Task type
    is_regression = (
        n_unique_target > 20 and
        profile["dtype_distribution"]["numeric"] > 0
    )

    if is_regression:
        task = "regression"
        reasons.append(f"Target has {n_unique_target} unique values — regression task detected")
    else:
        task = "classification"
        reasons.append(f"Target has {n_unique_target} unique values — classification task detected")

    # Model recommendations with weighted scoring
    model_scores = {}

    if task == "classification":
        model_scores["Random Forest"] = 0.7
        model_scores["Decision Tree"] = 0.5
        model_scores["Logistic Regression"] = 0.5
        model_scores["KNN"] = 0.4

        if n_rows > 10_000:
            model_scores["Random Forest"] += 0.2
            model_scores["KNN"] -= 0.3
            reasons.append("Large dataset — Random Forest scales well; KNN is slow")
        elif size_class == "tiny":
            model_scores["KNN"] += 0.2
            model_scores["Random Forest"] -= 0.1
            reasons.append("Small dataset — KNN and Decision Tree work well")

        if corr_density > 0.5:
            model_scores["Logistic Regression"] += 0.2
            reasons.append(f"High feature correlation density ({corr_density:.0%}) — linear models may suffice")

        if n_cols > 50:
            model_scores["Random Forest"] += 0.15
            model_scores["Decision Tree"] -= 0.1
            reasons.append("High dimensionality — Random Forest handles feature selection internally")

    else:  # regression
        model_scores["Linear Regression"] = 0.5
        model_scores["Random Forest"] = 0.7
        model_scores["Decision Tree"] = 0.4
        model_scores["KNN"] = 0.3

        if corr_density > 0.5:
            model_scores["Linear Regression"] += 0.25
            reasons.append("High correlation density — linear regression likely effective")

        if n_rows > 10_000:
            model_scores["KNN"] -= 0.3
            reasons.append("Large dataset — KNN regression is computationally expensive")

    # Sort and pick top 3
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    recommended = [m for m, _ in sorted_models[:3]]
    best_model = sorted_models[0][0]
    confidence = min(1.0, sorted_models[0][1])

    return {
        "decision": f"Primary: {best_model}",
        "confidence": round(confidence, 3),
        "reason": "; ".join(reasons),
        "recommended_models": recommended,
        "task_type": task,
        "all_scores": {m: round(s, 3) for m, s in sorted_models}
    }


def _decide_diagnostics(profile: dict) -> dict:
    checks = []
    reasons = []

    # Always run these
    checks.append("Missing Values")
    checks.append("Duplicate Rows")
    checks.append("Class Imbalance")

    if profile["overall_missing_ratio"] > 0.01:
        reasons.append(f"Missing values detected ({profile['overall_missing_ratio']:.1%}) — deep missing analysis enabled")

    if profile["dtype_distribution"]["numeric"] > 0:
        checks.append("Outlier Detection")
        checks.append("Feature Correlation")
        reasons.append("Numeric features present — outlier and correlation checks enabled")

    if profile["corr_density"] > 0.3:
        checks.append("Data Leakage Detection")
        reasons.append(f"High correlation density ({profile['corr_density']:.0%}) — leakage detection prioritised")

    if profile["n_rows"] > 500:
        checks.append("Label Noise Detection")
        reasons.append("Sufficient rows for KNN-based label noise detection")
    else:
        reasons.append("Small dataset — label noise detection skipped (requires ≥500 rows)")

    if profile["dtype_distribution"]["categorical"] > 0:
        checks.append("Mixed Field Inconsistencies")
        reasons.append("Categorical columns present — mixed field check enabled")

    return {
        "decision": f"Run {len(checks)} diagnostic checks",
        "confidence": 0.95,
        "reason": "; ".join(reasons),
        "checks": checks
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPLANATION LAYER
# ─────────────────────────────────────────────────────────────────────────────

def build_explanation_panel(profile: dict, strategies: dict) -> list:
    """
    Build a list of explanation cards for the UI's Decision Transparency Panel.

    Each card:
    {
        "category": "Imputation",
        "icon": "fa-fill-drip",
        "decision": "Median Imputation",
        "confidence": 0.87,
        "confidence_pct": 87,
        "reason": "...",
        "detail": "..."   # optional extra detail
    }
    """
    icon_map = {
        "imputation": ("fa-fill-drip", "Imputation Strategy"),
        "outlier_handling": ("fa-filter", "Outlier Handling"),
        "encoding": ("fa-tags", "Categorical Encoding"),
        "scaling": ("fa-ruler-combined", "Feature Scaling"),
        "imbalance": ("fa-balance-scale", "Class Imbalance"),
        "model_selection": ("fa-brain", "Model Selection"),
        "diagnostics_focus": ("fa-stethoscope", "Diagnostics Scope"),
    }

    cards = []
    for key, strategy in strategies.items():
        icon, label = icon_map.get(key, ("fa-cog", key.replace("_", " ").title()))
        card = {
            "category": label,
            "icon": icon,
            "decision": strategy["decision"],
            "confidence": strategy["confidence"],
            "confidence_pct": int(strategy["confidence"] * 100),
            "reason": strategy["reason"],
        }
        # Add recommended models list if present
        if "recommended_models" in strategy:
            card["detail"] = "Recommended: " + ", ".join(strategy["recommended_models"])
        if "checks" in strategy:
            card["detail"] = "Checks: " + ", ".join(strategy["checks"])
        cards.append(card)

    return cards


def run_intelligent_analysis(df: pd.DataFrame, target_col: str) -> dict:
    """
    Convenience wrapper: profile → strategies → explanation panel.
    Returns everything needed for the UI and pipeline.
    """
    profile = profile_dataset(df, target_col)
    strategies = select_strategies(profile)
    explanation_panel = build_explanation_panel(profile, strategies)

    return {
        "profile": profile,
        "strategies": strategies,
        "explanation_panel": explanation_panel
    }
