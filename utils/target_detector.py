"""
ADIE — Target Column Detector
Replaces the brittle "last column = target" assumption with an intelligent
ranked scoring system that evaluates every column as a candidate target.
"""

import pandas as pd
import numpy as np
import re


# Keywords that strongly suggest a column is a target/label
_TARGET_KEYWORDS = [
    'target', 'label', 'outcome', 'output', 'class', 'category',
    'result', 'response', 'dependent', 'y', 'predict', 'prediction',
    'diagnosis', 'status', 'flag', 'churn', 'default', 'fraud',
    'survived', 'survival', 'disease', 'risk', 'grade', 'score',
    'approved', 'rejected', 'pass', 'fail', 'decision', 'verdict',
    'type', 'group', 'cluster', 'segment', 'tag'
]

# Keywords that strongly suggest a column is NOT a target (identifier / feature)
_NON_TARGET_KEYWORDS = [
    'id', '_id', 'code', 'number', 'no', 'num', 'key', 'index',
    'name', 'first', 'last', 'email', 'phone', 'address', 'zip',
    'timestamp', 'date', 'time', 'created', 'updated', 'modified',
    'uuid', 'guid', 'hash', 'token', 'url', 'link', 'path',
    'description', 'comment', 'note', 'remark', 'text', 'body'
]


def detect_target_column(df: pd.DataFrame) -> dict:
    """
    Rank every column as a candidate target and return the best guess
    along with a full ranked list and per-column explanations.

    Returns
    -------
    {
        "recommended": "column_name",
        "confidence": 0.82,
        "reason": "Human-readable explanation",
        "candidates": [
            {"column": "...", "score": 0.82, "reason": "..."},
            ...
        ]
    }
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    candidates = []
    n_rows = len(df)

    for col in df.columns:
        score, reasons = _score_column(df[col], col, n_rows)
        candidates.append({
            "column": col,
            "score": round(score, 4),
            "reason": "; ".join(reasons) if reasons else "No strong signal"
        })

    # Sort descending by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    best = candidates[0] if candidates else {"column": df.columns[-1], "score": 0.0, "reason": "Fallback to last column"}

    return {
        "recommended": best["column"],
        "confidence": best["score"],
        "reason": best["reason"],
        "candidates": candidates
    }


def _score_column(series: pd.Series, col_name: str, n_rows: int) -> tuple:
    """
    Score a single column as a target candidate.
    Returns (score: float, reasons: list[str])
    Score is in [0, 1] — higher means more likely to be the target.
    """
    score = 0.0
    reasons = []
    col_lower = col_name.lower().strip()

    # ── 1. Name-based scoring ──────────────────────────────────────────────
    # Exact or partial match with known target keywords
    for kw in _TARGET_KEYWORDS:
        if col_lower == kw:
            score += 0.40
            reasons.append(f"Column name exactly matches target keyword '{kw}'")
            break
        if kw in col_lower:
            score += 0.20
            reasons.append(f"Column name contains target keyword '{kw}'")
            break

    # Penalise identifier-like names
    for kw in _NON_TARGET_KEYWORDS:
        if col_lower == kw or col_lower.endswith(kw) or col_lower.startswith(kw):
            score -= 0.30
            reasons.append(f"Column name suggests identifier/feature ('{kw}')")
            break

    # ── 2. Cardinality scoring ─────────────────────────────────────────────
    n_unique = series.nunique()
    unique_ratio = n_unique / n_rows if n_rows > 0 else 1.0

    if n_unique == 2:
        score += 0.25
        reasons.append("Binary column — ideal for classification target")
    elif 3 <= n_unique <= 20:
        score += 0.18
        reasons.append(f"Low cardinality ({n_unique} unique values) — good classification target")
    elif 21 <= n_unique <= 50:
        score += 0.08
        reasons.append(f"Moderate cardinality ({n_unique} unique values) — possible regression/multi-class target")
    elif unique_ratio > 0.9:
        score -= 0.25
        reasons.append(f"Very high cardinality ({unique_ratio:.0%} unique) — likely an identifier, not a target")

    # ── 3. Data type scoring ───────────────────────────────────────────────
    if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
        score += 0.10
        reasons.append("Categorical dtype — common for classification targets")
    elif pd.api.types.is_bool_dtype(series):
        score += 0.20
        reasons.append("Boolean dtype — strong signal for binary classification target")
    elif pd.api.types.is_integer_dtype(series) and n_unique <= 20:
        score += 0.12
        reasons.append("Integer with low cardinality — likely encoded class label")

    # ── 4. Missing value penalty ───────────────────────────────────────────
    missing_ratio = series.isnull().sum() / n_rows if n_rows > 0 else 0
    if missing_ratio > 0.30:
        score -= 0.20
        reasons.append(f"High missing rate ({missing_ratio:.0%}) — unreliable as target")
    elif missing_ratio > 0.05:
        score -= 0.05
        reasons.append(f"Some missing values ({missing_ratio:.0%})")

    # ── 5. Value distribution scoring ─────────────────────────────────────
    # Balanced classes are a good sign for a classification target
    if n_unique >= 2 and n_unique <= 20:
        try:
            counts = series.value_counts(normalize=True)
            max_share = counts.iloc[0]
            if max_share < 0.95:
                score += 0.08
                reasons.append(f"Reasonably balanced distribution (dominant class: {max_share:.0%})")
            else:
                score -= 0.05
                reasons.append(f"Heavily skewed distribution (dominant class: {max_share:.0%})")
        except Exception:
            pass

    # ── 6. Position bonus (last column convention) ────────────────────────
    # Mild bonus — we don't want to rely on this but it's a weak prior
    # (handled externally by the caller if needed)

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    return score, reasons


def get_column_preview(df: pd.DataFrame, max_rows: int = 5) -> list:
    """
    Return the first `max_rows` rows as a list of dicts for JSON serialisation.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Replace NaN with None for JSON compatibility
    preview = df.head(max_rows).where(pd.notnull(df.head(max_rows)), None)
    return preview.to_dict(orient='records')
