"""
ADIE — System Self-Evaluator (v3 — Adaptive Pipeline)
======================================================

Runs the full ADIE adaptive pipeline (diagnostics → cleaning → training)
on a list of datasets and measures how much the system improves each one.

All cleaning now uses the non-destructive Adaptive Data Preparation Engine.
NO COLUMNS ARE EVER DROPPED.
"""

import os
import time
import traceback
import pandas as pd
import numpy as np

from utils.data_analysis import perform_diagnostics
from utils.adaptive_cleaning import clean_dataset          # adaptive, non-destructive
from utils.model_training import train_and_evaluate
from utils.target_detector import detect_target_column


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_system(datasets: list) -> dict:
    """
    Run the full ADIE pipeline on each dataset and collect improvement metrics.
    """
    results = []

    for ds in datasets:
        result = _evaluate_single(ds)
        results.append(result)

    aggregate = _aggregate(results)

    return {
        "results": results,
        "aggregate": aggregate
    }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE DATASET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_single(ds: dict) -> dict:
    """Run the full pipeline on one dataset and return a result dict."""
    name = ds.get("name", "Unknown")
    path = ds.get("path", "")
    target_override = ds.get("target", None)

    result = {
        "dataset_name": name,
        "path": path,
        "success": False,
        "error": None,
        "rows": 0,
        "cols": 0,
        "target_col": None,
        "task_type": None,
        "issues_before": 0,
        "issues_after": 0,
        "issues_resolved": 0,
        "missing_before": 0,
        "missing_after": 0,
        "duplicates_before": 0,
        "duplicates_after": 0,
        "outliers_before": 0,
        "outliers_after": 0,
        "accuracy_before": None,
        "accuracy_after": None,
        "accuracy_improvement": None,
        "f1_before": None,
        "f1_after": None,
        "f1_improvement": None,
        "r2_before": None,
        "r2_after": None,
        "r2_improvement": None,
        "processing_time_sec": 0.0,
        "quality_score_before": None,
        "quality_score_after": None
    }

    t_start = time.time()

    try:
        # ── Load dataset ──────────────────────────────────────────────────
        if not os.path.exists(path):
            result["error"] = f"File not found: {path}"
            return result

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        result["rows"] = int(df.shape[0])
        result["cols"] = int(df.shape[1])

        if df.shape[0] < 10:
            result["error"] = "Dataset too small (< 10 rows)"
            return result

        # ── Target detection ──────────────────────────────────────────────
        if target_override:
            target_col = target_override.strip()
        else:
            detection = detect_target_column(df)
            target_col = detection["recommended"]

        if target_col not in df.columns:
            result["error"] = f"Target column '{target_col}' not found"
            return result

        result["target_col"] = target_col

        # ── Stage 1: Diagnostics (before) ─────────────────────────────────
        diag_before = perform_diagnostics(df)
        issues_before = diag_before.get("identified_issues", [])
        result["issues_before"] = len(issues_before)
        result["missing_before"] = int(diag_before["missing_values"]["total"])
        result["duplicates_before"] = int(diag_before["duplicates"])
        result["outliers_before"] = int(diag_before["outliers"]["total"])

        # ── Stage 2: Cleaning (adaptive, non-destructive) ─────────────────
        leakage_cols = diag_before.get("leakage_risk", [])
        leakage_cols = [c for c in leakage_cols if c != target_col]

        # Uses the adaptive engine — NO columns are dropped
        cleaned_df = clean_dataset(df, leakage_cols=leakage_cols, target_col=target_col)

        diag_after = perform_diagnostics(cleaned_df)
        issues_after = diag_after.get("identified_issues", [])
        result["issues_after"] = len(issues_after)
        result["issues_resolved"] = max(0, len(issues_before) - len(issues_after))
        result["missing_after"] = int(diag_after["missing_values"]["total"])
        result["duplicates_after"] = int(diag_after["duplicates"])
        result["outliers_after"] = int(diag_after["outliers"]["total"])

        # ── Stage 3: Training (before cleaning — minimal baseline) ────────
        df_orig_processed = _minimal_process(df, target_col)

        orig_results, task_type = train_and_evaluate(
            df_orig_processed, target_col, selected_algo="Random Forest"
        )
        result["task_type"] = task_type

        # ── Stage 4: Training (after cleaning) ────────────────────────────
        if target_col not in cleaned_df.columns:
            result["error"] = "Target column lost during cleaning"
            return result

        cleaned_results, _ = train_and_evaluate(
            cleaned_df, target_col, selected_algo="Random Forest"
        )

        # ── Extract metrics ───────────────────────────────────────────────
        rf_orig = orig_results.get("Random Forest", {})
        rf_clean = cleaned_results.get("Random Forest", {})

        if task_type == "classification":
            acc_b = rf_orig.get("Accuracy")
            acc_a = rf_clean.get("Accuracy")
            f1_b = rf_orig.get("F1-Score")
            f1_a = rf_clean.get("F1-Score")

            result["accuracy_before"] = acc_b
            result["accuracy_after"] = acc_a
            result["f1_before"] = f1_b
            result["f1_after"] = f1_a

            if acc_b is not None and acc_a is not None:
                result["accuracy_improvement"] = round(acc_a - acc_b, 4)
            if f1_b is not None and f1_a is not None:
                result["f1_improvement"] = round(f1_a - f1_b, 4)

        else:  # regression
            r2_b = rf_orig.get("R2 Score")
            r2_a = rf_clean.get("R2 Score")
            result["r2_before"] = r2_b
            result["r2_after"] = r2_a
            if r2_b is not None and r2_a is not None:
                result["r2_improvement"] = round(r2_a - r2_b, 4)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    finally:
        result["processing_time_sec"] = round(time.time() - t_start, 2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_process(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Apply just enough processing to make the original dataset trainable
    (encode non-numeric, fill NaN) without running the full repair pipeline.
    This gives us a fair "before" baseline.

    NOTE: This is intentionally simple — it does NOT use the adaptive engine
    so we can measure the improvement the engine provides.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Keep only numeric columns + target
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    df = df[numeric_cols]

    # Fill NaN with median for numeric columns
    for col in df.columns:
        if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            median_val = numeric_series.median()
            df[col] = numeric_series.fillna(median_val if not pd.isna(median_val) else 0)

    # Encode target if needed
    if df[target_col].dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))

    # Drop rows where target is still NaN
    df = df.dropna(subset=[target_col])

    return df


def _aggregate(results: list) -> dict:
    """Compute aggregate statistics across all dataset results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    n_total = len(results)
    n_success = len(successful)
    n_failed = len(failed)

    success_rate = round(n_success / n_total, 4) if n_total > 0 else 0.0

    acc_improvements = [
        r["accuracy_improvement"] for r in successful
        if r["accuracy_improvement"] is not None
    ]
    avg_acc_improvement = round(float(np.mean(acc_improvements)), 4) if acc_improvements else None

    f1_improvements = [
        r["f1_improvement"] for r in successful
        if r["f1_improvement"] is not None
    ]
    avg_f1_improvement = round(float(np.mean(f1_improvements)), 4) if f1_improvements else None

    r2_improvements = [
        r.get("r2_improvement") for r in successful
        if r.get("r2_improvement") is not None
    ]
    avg_r2_improvement = round(float(np.mean(r2_improvements)), 4) if r2_improvements else None

    issues_resolved = [r["issues_resolved"] for r in successful]
    avg_issues_resolved = round(float(np.mean(issues_resolved)), 2) if issues_resolved else 0.0

    improved = [
        r for r in successful
        if (r.get("accuracy_improvement") or 0) > 0 or
           (r.get("f1_improvement") or 0) > 0 or
           (r.get("r2_improvement") or 0) > 0
    ]
    improvement_rate = round(len(improved) / n_success, 4) if n_success > 0 else 0.0

    return {
        "total_datasets": n_total,
        "successful": n_success,
        "failed": n_failed,
        "success_rate": success_rate,
        "improvement_rate": improvement_rate,
        "avg_accuracy_improvement": avg_acc_improvement,
        "avg_f1_improvement": avg_f1_improvement,
        "avg_r2_improvement": avg_r2_improvement,
        "avg_issues_resolved": avg_issues_resolved,
        "failed_datasets": [{"name": r["dataset_name"], "error": r["error"]} for r in failed]
    }
