# ADIE v2.0 — Full Project Understanding Document
> **For use with ChatGPT or any AI assistant for further development, debugging, or extension.**
> **Updated: April 2026 — Post-Upgrade (Intelligent Engine + Target Detection + System Evaluation + UI Overhaul)**

---

## 1. What Is This Project?

**ADIE** stands for **Automated Dataset Intelligence Engine**.

It is a **Flask-based web application** that acts as an end-to-end, intelligent ML data preparation platform. A user uploads a CSV or ZIP dataset, and ADIE automatically:

1. **Detects** the most likely target column (or lets the user choose)
2. **Profiles** the dataset and makes data-driven strategy decisions with confidence scores
3. **Diagnoses** 9 categories of data quality issues
4. **Repairs** the dataset through an adaptive cleaning pipeline
5. **Trains and benchmarks** multiple ML algorithms on both original and cleaned data
6. **Explains** every decision it made (why this imputation? why this model?)
7. **Evaluates itself** across multiple datasets to prove its effectiveness
8. **Generates** a downloadable professional PDF report

The platform is designed for **data scientists, researchers, and final-year project submissions** who want to quickly assess and prepare a dataset for machine learning without writing code.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask 2.3+ (Python) |
| Data Processing | pandas 2.0+, numpy 1.24+, scipy 1.11+ |
| Machine Learning | scikit-learn 1.3+ |
| Class Imbalance | imbalanced-learn 0.11+ (SMOTE) |
| Model Serialization | joblib 1.3+ |
| PDF Reports | reportlab 4.0+ |
| Frontend Templating | Jinja2 (built into Flask) |
| Frontend Styling | Custom CSS (no framework) |
| Charts | Chart.js (CDN) |
| Icons | Font Awesome 6.4 (CDN) |
| Fonts | Google Fonts: Plus Jakarta Sans, Space Grotesk |
| Visualization (optional) | matplotlib, seaborn |

---

## 3. Project File Structure

```
AI-Project/
+-- app.py                          # Main Flask application (16 routes)
+-- requirements.txt                # Python dependencies
+-- FINAL REPORT.docx               # Academic/project report document
+-- PROJECT_UNDERSTANDING.md        # This file
|
+-- utils/                          # Backend logic modules (7 files)
|   +-- column_detector.py          # Classifies columns by type
|   +-- data_analysis.py            # 9-point diagnostic engine
|   +-- data_cleaning.py            # Adaptive cleaning pipeline
|   +-- model_training.py           # Multi-algorithm training + evaluation
|   +-- dataset_expert.py           # SWOT, domain detection, interventions
|   +-- target_detector.py          # NEW: Intelligent target column detection
|   +-- intelligent_engine.py       # NEW: Data-driven strategy selector
|   +-- system_evaluator.py         # NEW: Self-evaluation across datasets
|   +-- report_generator.py         # UPGRADED: PDF + text report generation
|
+-- templates/                      # HTML pages (Jinja2)
|   +-- splash.html                 # Landing page (auto-redirects)
|   +-- auth.html                   # Login + Signup
|   +-- index.html                  # UPGRADED: Dashboard with drag-drop + target selector
|   +-- result.html                 # UPGRADED: Full analysis dashboard with 7 sections
|
+-- static/
|   +-- style.css                   # UPGRADED: All CSS with new components
|
+-- data/default/                   # Pre-loaded datasets
|   +-- Data.Gov+-+FY25+Q4.csv
|   +-- archive.zip
|
+-- uploads/                        # Runtime storage (generated during use)
|   +-- current_dataset.csv         # Active dataset
|   +-- cleaned_dataset.csv         # Post-repair dataset
|   +-- metadata.json               # Dataset metadata + target_col
|   +-- diagnostics.json            # Diagnostic results + target_col
|   +-- original_diagnostics.json   # Pre-repair diagnostics
|   +-- expert_report.json          # SWOT + suitability
|   +-- intelligent_analysis.json   # NEW: Profile + strategies + explanations
|   +-- ml_results.json             # Model training results
|   +-- eval_results.json           # NEW: System evaluation results
|   +-- analysis_report.pdf         # UPGRADED: PDF report (was .txt)
|   +-- best_model.pkl              # Best trained model
|   +-- scaler.pkl                  # StandardScaler
|   +-- encoder_mappings.pkl        # All encoders
|   +-- users.json                  # User credentials
|   +-- version_YYYYMMDD_HHMMSS/    # Versioned snapshots
```

---

## 4. All Routes (16 total)

| Route | Method | Auth | Description |
|---|---|---|---|
| `/` | GET | No | Splash screen |
| `/login` | GET | No | Login page |
| `/signup` | GET | No | Signup page |
| `/login_post` | POST | No | Validate credentials |
| `/signup_post` | POST | No | Create account |
| `/logout` | GET | No | Clear session |
| `/dashboard` | GET | Yes | Main dashboard with upload zone |
| `/preview` | POST | Yes | **NEW** AJAX: returns columns, preview rows, target detection |
| `/analyze` | POST | Yes | Upload + Stage 1 diagnostics + intelligent analysis |
| `/analyze_default` | POST | Yes | Same pipeline for pre-loaded datasets |
| `/clean` | POST | Yes | Stage 2: adaptive cleaning |
| `/train` | POST | Yes | Stage 3: model training + benchmarking |
| `/evaluate` | POST | Yes | **NEW** System self-evaluation (returns JSON) |
| `/download_cleaned` | GET | Yes | Download cleaned CSV |
| `/download_report` | GET | Yes | **UPGRADED** Generate + download PDF report |
| `/static/<path>` | GET | No | Static file serving |

---

## 5. The Four-Stage ADIE Pipeline

### Stage 0 --- Target Detection (NEW)

**Trigger**: Immediately after file upload (via AJAX `/preview` endpoint)
**Module**: `utils/target_detector.py`

The system scores every column across 5 dimensions:
1. **Name keywords** (+0.40 for exact match like "target", "label", "class"; -0.30 for "id", "name", "date")
2. **Cardinality** (+0.25 for binary; +0.18 for 3-20 unique; -0.25 for >90% unique)
3. **Data type** (+0.20 for boolean; +0.12 for low-cardinality integer; +0.10 for categorical)
4. **Missing values** (-0.20 if >30% missing)
5. **Value distribution** (+0.08 if reasonably balanced)

Returns:
```json
{
  "recommended": "future_diabetes_5yr",
  "confidence": 0.85,
  "reason": "Column name contains target keyword 'future'; Binary column",
  "candidates": [...]
}
```

The user can override via a dropdown or accept the auto-detection.

---

### Stage 1 --- Diagnostics + Intelligent Analysis

**Route**: `/analyze` or `/analyze_default`
**Modules**: `data_analysis.py`, `dataset_expert.py`, `intelligent_engine.py`

Steps:
1. Load CSV, sanitize column names
2. Validate target column (from user selection or auto-detect)
3. Detect column types (`column_detector.py`)
4. Extract metadata (rows, cols, size, dtypes, target_col)
5. Run `perform_diagnostics()` --- 9 checks
6. Run `analyze_dataset_expertly()` --- domain, SWOT, interventions
7. **NEW**: Run `run_intelligent_analysis()`:
   - Profile the dataset (size class, missing ratio, skewness, correlation density, cardinality, etc.)
   - Select strategies for: imputation, outlier handling, encoding, scaling, imbalance, model selection, diagnostics scope
   - Build explanation panel (7 cards with icons, confidence bars, reasons)
8. Save all JSON artefacts
9. Render `result.html` with all data

---

### Stage 2 --- Adaptive Cleaning

**Route**: `/clean`
**Module**: `data_cleaning.py`

Steps (in order):
1. Remove identifier columns (never drop target)
2. Remove leakage columns (>95% correlation with target)
3. Remove duplicates
4. Drop columns with >90% missing (never drop target)
5. Parse datetime columns into year/month/day/dayofweek features
6. Handle mixed-type columns (coerce to numeric)
7. Impute missing values (median for numeric, mode for categorical)
8. Cap outliers (IQR method)
9. Encode categoricals:
   - Low cardinality (<=50 unique): OneHotEncoder
   - High cardinality (>50 unique): Frequency encoding
   - Ordinal: OrdinalEncoder
   - Target: LabelEncoder
10. Create versioned backup with before/after CSVs
11. Re-run diagnostics on cleaned data
12. Calculate improvements (issues resolved, rows/columns removed)

---

### Stage 3 --- Model Training + Benchmarking

**Route**: `/train`
**Module**: `model_training.py`

Steps:
1. Detect task type (classification if target <=20 unique values; else regression)
2. Split 80/20 with stratification
3. Apply SMOTE if classification + imbalanced
4. Scale with StandardScaler
5. Train models:
   - Classification: Random Forest, Decision Tree, Logistic Regression, KNN
   - Regression: Linear Regression, Random Forest, Decision Tree, KNN
6. Evaluate: Accuracy/Precision/Recall/F1 (classification) or R2/MAE/MSE (regression)
7. Extract feature importance from tree-based models
8. Save best model + scaler
9. Train on BOTH original (processed) and cleaned datasets for comparison
10. Store results in `ml_results.json`

---

## 6. NEW: Intelligent Decision Engine (`utils/intelligent_engine.py`)

### Dataset Profiler

`profile_dataset(df, target_col)` produces:
```json
{
  "n_rows": 496362,
  "n_cols": 35,
  "size_class": "large",
  "overall_missing_ratio": 0.0012,
  "dtype_distribution": {"numeric": 28, "categorical": 7, "boolean": 0},
  "cardinality": {"country": 45, "gender": 2, ...},
  "high_cardinality_cols": ["country"],
  "skewness": {"age": 0.12, "BMI": 0.85, ...},
  "highly_skewed_cols": ["insulin", "HOMA_IR"],
  "avg_skewness": 0.45,
  "corr_density": 0.15,
  "high_corr_pairs": [{"col_a": "...", "col_b": "...", "correlation": 0.92}],
  "target_info": {"unique_values": 2, "imbalance_ratio": 3.5, "dominant_class_share": 0.78},
  "n_duplicates": 0,
  "duplicate_ratio": 0.0
}
```

### Strategy Selector

`select_strategies(profile)` returns 7 decisions:

| Strategy | Example Output |
|---|---|
| Imputation | `{decision: "Median Imputation", confidence: 0.87, reason: "..."}` |
| Outlier Handling | `{decision: "IQR Capping", confidence: 0.80, reason: "..."}` |
| Encoding | `{decision: "Frequency + One-Hot", confidence: 0.90, reason: "..."}` |
| Scaling | `{decision: "StandardScaler", confidence: 0.88, reason: "..."}` |
| Imbalance | `{decision: "SMOTE Oversampling", confidence: 0.80, reason: "..."}` |
| Model Selection | `{decision: "Primary: Random Forest", confidence: 0.90, recommended_models: [...]}` |
| Diagnostics Scope | `{decision: "Run 7 checks", confidence: 0.95, checks: [...]}` |

Each decision uses **weighted scoring** (not simple if-else). Multiple options are scored and the highest wins.

### Explanation Panel

`build_explanation_panel(profile, strategies)` returns a list of UI cards:
```json
[
  {
    "category": "Imputation Strategy",
    "icon": "fa-fill-drip",
    "decision": "Median Imputation",
    "confidence": 0.87,
    "confidence_pct": 87,
    "reason": "Very low missing rate (0.1%); High skewness (1.2) - median preferred"
  },
  ...
]
```

---

## 7. NEW: System Self-Evaluator (`utils/system_evaluator.py`)

### Purpose
Proves ADIE's effectiveness by running the full pipeline on multiple datasets and measuring improvement.

### Function: `evaluate_system(datasets)`

For each dataset:
1. Load CSV
2. Auto-detect target column
3. Run diagnostics (before)
4. Run full cleaning pipeline
5. Run diagnostics (after)
6. Train Random Forest on original (minimally processed)
7. Train Random Forest on cleaned
8. Collect: accuracy before/after, F1 before/after, issues resolved, processing time

### Output Structure
```json
{
  "results": [
    {
      "dataset_name": "diabetes.csv",
      "success": true,
      "rows": 496362,
      "target_col": "future_diabetes_5yr",
      "task_type": "classification",
      "issues_before": 5,
      "issues_after": 0,
      "issues_resolved": 5,
      "accuracy_before": 0.82,
      "accuracy_after": 0.94,
      "accuracy_improvement": 0.12,
      "f1_before": 0.79,
      "f1_after": 0.91,
      "f1_improvement": 0.12,
      "processing_time_sec": 45.2
    },
    ...
  ],
  "aggregate": {
    "total_datasets": 5,
    "successful": 4,
    "failed": 1,
    "success_rate": 0.80,
    "improvement_rate": 0.75,
    "avg_accuracy_improvement": 0.08,
    "avg_f1_improvement": 0.07,
    "avg_issues_resolved": 3.5
  }
}
```

### UI
- Triggered by "RUN SYSTEM EVALUATION" button on result page
- Shows: summary cards, bar chart (accuracy before/after per dataset), full results table with status badges

---

## 8. Diagnostics --- 9 Checks (`data_analysis.py`)

| # | Check | Method | What It Detects |
|---|---|---|---|
| 1 | Missing Values | `isnull().sum()` | Count per column + total |
| 2 | Duplicate Rows | `duplicated().sum()` | Exact duplicate rows |
| 3 | Basic Statistics | `describe()` | Mean, median, std per numeric column |
| 4 | Outliers | IQR method (1.5xIQR) | Extreme values per numeric column |
| 5 | Class Imbalance | `value_counts()` on target | Distribution of target classes |
| 6 | Feature Correlation / Leakage | Pearson correlation | Features with >95% correlation flagged |
| 7 | Label Noise | KNN-based semantic analysis | Samples likely mislabelled |
| 8 | Mixed Field Inconsistencies | Type inspection | Columns with mixed numeric/text |
| 9 | Issue Identification | Aggregated from above | Severity-scored list of all issues |

---

## 9. Column Type Detection (`column_detector.py`)

| Type | Description | Examples |
|---|---|---|
| `identifiers` | ID/code columns (removed before training) | `patient_id`, `order_id` |
| `datetime_cols` | Date/time columns (parsed into features) | `created_at`, `date` |
| `numerical_cols` | Continuous or discrete numbers | `age`, `price`, `BMI` |
| `nominal_categorical` | Categories with no natural order | `gender`, `country` |
| `ordinal_categorical` | Categories with a natural order | `rating`, `education_level` |

---

## 10. Expert Analysis (`dataset_expert.py`)

Produces:
```json
{
  "summary": {
    "rows": 496362, "cols": 39,
    "domain": "Healthcare / Clinical",
    "industry": "Medical Research",
    "quality_score": 100.0,
    "suitability": "HIGHLY SUITABLE (REPAIRED)",
    "version": "Repaired"
  },
  "swot": { "strengths": [...], "weaknesses": [...] },
  "interventions": [
    {"issue": "Class Imbalance", "strategy": "SMOTE Resampling", "rationale": "..."}
  ],
  "tasks": ["Binary Classification", "Regression Analysis"],
  "recommendations": [...]
}
```

Domain detection is keyword-based (column names scanned for healthcare/finance/marketing terms).

---

## 11. Frontend Pages (4 templates)

### `splash.html` --- Landing Page
- Animated ADIE logo, auto-redirects to login after 3 seconds

### `auth.html` --- Login/Signup
- Shared template toggled by `signup` boolean
- Flash messages for errors

### `index.html` --- Dashboard (UPGRADED)
- **Drag-and-drop upload zone** with visual feedback (icon changes, border colour)
- **Live file preview** via AJAX `/preview` endpoint:
  - Shows file name, size, rows x columns
  - Populates target column dropdown with all columns + recommended star
  - Renders first 5 rows in a scrollable table
- **Target column selector**: dropdown with "Auto-detect (recommended)" default
- **Default datasets card**: dropdown + submit
- **Feature cards**: Diagnostics, Intelligent Engine, Repair, ML Benchmark

### `result.html` --- Analysis Dashboard (UPGRADED - 7 sections)

| Section | When Shown | Content |
|---|---|---|
| 1. Expert Report | Always (after Stage 1) | Quality score, domain, industry, suitability badge, issues grid, interventions, SWOT |
| 2. Decision Transparency Panel | Always (after Stage 1) | **NEW** 7 expandable cards showing every strategy decision with confidence bars and reasoning |
| 3. Metadata | Always | Rows, columns, size, export button |
| 4. Pipeline Control | Always | Execute Pipeline button (Stage 1->2) or Algorithm selector + Run Benchmark (Stage 2->3) |
| 5. Before vs After Analytics | After cleaning | **UPGRADED** Summary cards, 2 charts (issue severity + data quality metrics), side-by-side issue lists, improvement indicators |
| 6. ML Performance Benchmarks | After training | **UPGRADED** Toggle original/cleaned, animated metric cards, comparison chart, feature importance chart, prediction trend chart, PDF download button |
| 7. System Evaluation Dashboard | After training | **NEW** Run Evaluation button, aggregate summary cards, bar chart, full results table |

---

## 12. Authentication System

- **Storage**: `uploads/users.json` (flat JSON: `{"username": "password"}`)
- **Default user**: `admin` / `password123`
- **Session**: Flask server-side session (`session['user']`)
- **Secret key**: `'supersecretkey'` (hardcoded --- not production-safe)
- **Protection**: `@login_required` decorator on all protected routes
- **Passwords**: Plaintext (demo/academic project)

---

## 13. Versioning System

Every `/clean` call creates:
```
uploads/version_20260425_000650/
+-- before_clean.csv
+-- after_clean.csv
+-- version_info.json
```

`version_info.json`:
```json
{
  "timestamp": "20260425_000650",
  "original_rows": 1000, "cleaned_rows": 950,
  "original_columns": 20, "cleaned_columns": 17,
  "original_issues": 5, "cleaned_issues": 1,
  "improvements": {
    "issues_resolved": 4,
    "resolved_list": ["Missing Values", "Duplicates", "Outliers", "Data Leakage"],
    "rows_removed": 50, "columns_removed": 3
  }
}
```

---

## 14. Model Persistence Files

| File | Contents | Used For |
|---|---|---|
| `best_model.pkl` | Trained sklearn estimator | Future predictions |
| `scaler.pkl` | Fitted StandardScaler | Normalising new data |
| `encoder_mappings.pkl` | OneHot/Ordinal/Label encoders | Encoding new data |
| `encoder_mappings_info.pkl` | JSON-serializable encoder metadata | Inspection |

---

## 15. PDF Report Generation (`utils/report_generator.py`)

### `generate_pdf_report()` (NEW)
Uses **reportlab** to produce a professional A4 PDF with:
- Cover block (dark green gradient header)
- Executive Summary table (version, suitability, quality score, domain, size)
- SWOT analysis (green/red split table)
- Issue Identification table (type, severity badge, score)
- Intervention Selection table (issue, strategy, rationale)
- Detailed Diagnostics (missing, duplicates, outliers, correlations)
- ML Performance Comparison (Original vs Repaired with delta columns showing improvement arrows)
- System Evaluation section (if available)
- Final Determination footer

Falls back to `.txt` if reportlab is not installed.

### `generate_text_report()` (Legacy)
Still available for backward compatibility.

---

## 16. Design Patterns

| Pattern | Where | Purpose |
|---|---|---|
| Pipeline Pattern | 4-stage flow (Detect -> Diagnose -> Repair -> Train) | Sequential, modular processing |
| Strategy Pattern | `intelligent_engine.py` weighted scoring | Data-driven decision selection |
| Decorator Pattern | `@login_required` | Route-level authentication |
| Factory Pattern | `get_models()` | Returns algorithm dict by task type |
| Observer Pattern | AJAX `/preview` endpoint | Real-time UI updates without page reload |
| Expert System | `dataset_expert.py` + `intelligent_engine.py` | Domain detection + rule-based recommendations |
| Self-Evaluation | `system_evaluator.py` | System proves its own effectiveness |

---

## 17. Key Architectural Decisions

1. **Target column is never assumed** --- detected via scoring or user-selected via dropdown
2. **Every decision is explainable** --- confidence score + human-readable reason
3. **Weighted scoring over if-else** --- multiple options scored, highest wins
4. **AJAX preview before submission** --- user sees data before committing to analysis
5. **System evaluates itself** --- runs pipeline on multiple datasets, reports aggregate improvement
6. **PDF over TXT** --- professional output suitable for academic submission
7. **Graceful fallbacks** --- PDF falls back to TXT; target detection falls back to last column
8. **All artefacts are JSON** --- human-readable, easy to parse, enables report generation
9. **Versioned snapshots** --- full audit trail with before/after CSVs
10. **Modular utils/** --- each concern in its own file, testable independently

---

## 18. Data Flow Summary

```
User selects file (drag-drop or click)
        |
        v
/preview (AJAX) --> target_detector.py --> returns columns + preview + target recommendation
        |
        v
User confirms target column (or accepts auto-detect)
        |
        v
/analyze --> column_detector.py --> data_analysis.py --> dataset_expert.py --> intelligent_engine.py
        |
        v
result.html (Stage 1: Diagnostics + Decision Transparency Panel)
        |
        v
User clicks "Execute Pipeline"
        |
        v
/clean --> data_cleaning.py --> version snapshot --> re-diagnose --> compare before/after
        |
        v
result.html (Stage 2: Before/After Analytics with charts + improvement indicators)
        |
        v
User selects algorithm, clicks "Run Benchmark"
        |
        v
/train --> model_training.py --> best_model.pkl + scaler.pkl
        |
        v
result.html (Stage 3: Performance charts + toggle original/cleaned + feature importance)
        |
        v
User clicks "Run System Evaluation"
        |
        v
/evaluate --> system_evaluator.py --> runs pipeline on all available datasets
        |
        v
result.html (System Evaluation Dashboard: summary cards + chart + table)
        |
        v
User clicks "Download PDF Report"
        |
        v
/download_report --> report_generator.py --> ADIE_Analysis_Report.pdf
```

---

## 19. Quick Reference --- All Key Functions

| Function | File | Purpose |
|---|---|---|
| `detect_target_column(df)` | `target_detector.py` | Score all columns, return best target candidate |
| `get_column_preview(df)` | `target_detector.py` | Return first 5 rows as JSON |
| `profile_dataset(df, target)` | `intelligent_engine.py` | Produce rich dataset profile |
| `select_strategies(profile)` | `intelligent_engine.py` | 7 weighted strategy decisions |
| `build_explanation_panel(...)` | `intelligent_engine.py` | UI cards with confidence + reasons |
| `run_intelligent_analysis(df, target)` | `intelligent_engine.py` | Convenience: profile + strategies + panel |
| `evaluate_system(datasets)` | `system_evaluator.py` | Run pipeline on multiple datasets, aggregate results |
| `perform_diagnostics(df)` | `data_analysis.py` | Run all 9 diagnostic checks |
| `analyze_dataset_expertly(df, diag)` | `dataset_expert.py` | SWOT, domain, quality score, interventions |
| `detect_column_types(df, target)` | `column_detector.py` | Classify columns by type |
| `clean_dataset(df, leakage, target)` | `data_cleaning.py` | Full adaptive cleaning pipeline |
| `detect_task_type(df, target)` | `model_training.py` | Return 'classification' or 'regression' |
| `get_models(task_type)` | `model_training.py` | Return dict of sklearn model instances |
| `train_and_evaluate(df, target, algo)` | `model_training.py` | Train, evaluate, save best model |
| `generate_pdf_report(...)` | `report_generator.py` | Professional PDF with all sections |
| `generate_text_report(...)` | `report_generator.py` | Legacy text report |
| `login_required(f)` | `app.py` | Decorator: redirect if not authenticated |
| `_run_stage1(csv_path, filename, target)` | `app.py` | Shared helper: load + diagnose + analyse |
| `_save_json(filename, data)` | `app.py` | Persist JSON to uploads/ |
| `_load_json(filename)` | `app.py` | Load JSON from uploads/ |

---

## 20. CSS Design System (`static/style.css`)

**Colour Palette:**
```css
--primary: #065f46;       /* Deep Forest Green */
--accent: #10b981;        /* Emerald Green */
--secondary: #334155;     /* Charcoal Grey */
--danger: #991b1b;        /* Deep Crimson */
--warning: #92400e;       /* Burnt Amber */
--bg-main: #f8fafc;       /* Light Greyish Blue */
--text-main: #1e293b;     /* Charcoal Navy */
```

**Typography:**
- Body: `Plus Jakarta Sans`
- Headings: `Space Grotesk`

**Key Components:**
- `.hero` --- gradient banner with left accent border
- `.card` --- white card with fadeInUp animation
- `.btn` --- uppercase button (primary/secondary)
- `.upload-zone` --- dashed drag-drop area with hover/drag-over states
- `.decision-card` --- expandable `<details>` element for transparency panel
- `.alert` --- info/success/warning with left border accent
- `.stat-label` / `.stat-value` --- large metric displays
- `#loadingOverlay` --- full-screen spinner with animated progress bar
- `.badge-improved` / `.badge-declined` --- green/red improvement indicators
- `#previewTable` --- scrollable data preview table
- `#evalTable` --- system evaluation results table

---

## 21. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask app
python app.py

# 3. Open in browser
# http://127.0.0.1:5000

# Default login: admin / password123
```

The app runs in debug mode with thread limits for memory efficiency:
```python
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
```

---

## 22. Dependencies (`requirements.txt`)

```
Flask>=2.3.0              # Web framework
pandas>=2.0.0             # Data manipulation
numpy>=1.24.0             # Numerical computing
scipy>=1.11.0             # Statistical functions (skewness, etc.)
scikit-learn>=1.3.0       # ML algorithms
imbalanced-learn>=0.11.0  # SMOTE for class imbalance
joblib>=1.3.0             # Model serialization
reportlab>=4.0.0          # PDF report generation
matplotlib>=3.7.0         # Visualization (optional)
seaborn>=0.12.0           # Statistical visualization (optional)
regex>=2023.0.0           # Advanced regex patterns
```

---

## 23. Known Limitations / Security Notes

1. Passwords stored in plaintext in `users.json`
2. Secret key is hardcoded (`'supersecretkey'`)
3. No file size limit on uploads
4. Single-user state (all users share `uploads/` folder)
5. No CSRF protection on forms
6. No input sanitisation on username/password
7. Thread safety issues with concurrent users
8. No `/predict` route (model trained but not exposed for inference)
9. System evaluation limited to 5 datasets per run (for performance)

---

## 24. What Makes This Academically Strong

| Feature | Academic Value |
|---|---|
| Intelligent Decision Engine | Demonstrates data-driven reasoning, not just hard-coded rules |
| Confidence Scores | Quantifies uncertainty in every decision |
| Explainability Panel | Shows the system can justify its actions (XAI principle) |
| System Self-Evaluation | Proves effectiveness with empirical evidence across datasets |
| Target Auto-Detection | Shows the system adapts to unknown datasets |
| Before/After Comparison | Demonstrates measurable improvement |
| PDF Report | Professional output suitable for submission |
| Modular Architecture | Clean separation of concerns, testable components |
| Versioning | Full audit trail and reproducibility |
| Multiple ML Algorithms | Comparative analysis, not single-model bias |

---

*This document was auto-generated from full codebase analysis of the ADIE v2.0 application (April 2026).*
