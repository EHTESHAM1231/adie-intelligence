# ADIE v2.0 — Complete Project Understanding Document
> **Purpose**: Give this entire file to ChatGPT (or any AI) so it fully understands the project for further development, debugging, or deployment help.
> **Last Updated**: April 2026 — Deployed on Render

---

## 1. PROJECT OVERVIEW

**Name**: ADIE (Automated Dataset Intelligence Engine)
**Type**: Flask web application (Python)
**Purpose**: End-to-end ML data preparation platform that:
- Auto-detects the target column (or lets user choose)
- Diagnoses 9 categories of data quality issues
- Makes intelligent, explainable strategy decisions (not hard-coded rules)
- Repairs datasets through adaptive cleaning
- Trains and benchmarks multiple ML algorithms
- Evaluates its own effectiveness across datasets
- Generates professional PDF reports

**Deployed at**: https://adie-intelligence.onrender.com (Render free tier)
**Login**: admin / password123

---

## 2. TECH STACK

| Component | Technology |
|---|---|
| Backend | Flask 2.3+ (Python 3.10+) |
| ML | scikit-learn 1.3+, imbalanced-learn (SMOTE) |
| Data | pandas 2.0+, numpy 1.24+, scipy 1.11+ |
| PDF Reports | reportlab 4.0+ |
| Production Server | Gunicorn |
| Hosting | Render.com (free tier) |
| Frontend | Jinja2 templates + custom CSS + Chart.js |
| Icons | Font Awesome 6.4 (CDN) |
| Fonts | Google Fonts: Plus Jakarta Sans, Space Grotesk |

---

## 3. FILE STRUCTURE

```
AI-Project/                         <-- THIS IS THE REPO ROOT
|-- app.py                          Main Flask app (all 16 routes)
|-- Procfile                        Render start command
|-- render.yaml                     Render auto-config
|-- requirements.txt                All Python dependencies
|-- .gitignore                      Excludes uploads, large files
|
|-- static/
|   +-- style.css                   All CSS (16KB, emerald/charcoal theme)
|
|-- templates/
|   |-- splash.html                 Landing page (auto-redirects to login)
|   |-- auth.html                   Login + Signup (shared template)
|   |-- index.html                  Dashboard (drag-drop upload + target selector)
|   +-- result.html                 Analysis dashboard (7 sections)
|
|-- utils/                          Backend modules (7 files)
|   |-- column_detector.py          Classifies columns by type
|   |-- data_analysis.py            9-point diagnostic engine
|   |-- data_cleaning.py            Adaptive cleaning pipeline
|   |-- model_training.py           Multi-algorithm training
|   |-- dataset_expert.py           SWOT, domain detection, interventions
|   |-- target_detector.py          Intelligent target column scoring
|   |-- intelligent_engine.py       Weighted strategy selector + explainer
|   |-- system_evaluator.py         Self-evaluation across datasets
|   +-- report_generator.py         PDF + text report generation
|
|-- data/default/                   Pre-loaded sample datasets
|   +-- archive.zip                 Small demo dataset (0.17MB)
|
|-- uploads/                        Runtime storage (NOT in git)
|   |-- current_dataset.csv         Active dataset being processed
|   |-- cleaned_dataset.csv         Post-repair dataset
|   |-- metadata.json               Dataset metadata + target_col
|   |-- diagnostics.json            Diagnostic results
|   |-- expert_report.json          SWOT + suitability
|   |-- intelligent_analysis.json   Profile + strategies + explanations
|   |-- ml_results.json             Model training results
|   |-- eval_results.json           System evaluation results
|   |-- best_model.pkl              Trained model
|   |-- scaler.pkl                  StandardScaler
|   |-- encoder_mappings.pkl        All encoders
|   |-- users.json                  User credentials
|   +-- version_YYYYMMDD_HHMMSS/    Versioned cleaning snapshots
```

---

## 4. ALL 16 ROUTES

| Route | Method | Auth | What It Does |
|---|---|---|---|
| `/` | GET | No | Splash page -> auto-redirects to /login after 3s |
| `/login` | GET | No | Login form |
| `/signup` | GET | No | Signup form |
| `/login_post` | POST | No | Validates credentials, sets session |
| `/signup_post` | POST | No | Creates user, sets session |
| `/logout` | GET | No | Clears session |
| `/dashboard` | GET | Yes | Upload zone + default datasets |
| `/preview` | POST | Yes | AJAX: returns columns, 5-row preview, target detection |
| `/analyze` | POST | Yes | Upload file -> Stage 1 diagnostics + intelligent analysis |
| `/analyze_default` | POST | Yes | Same pipeline for pre-loaded datasets |
| `/clean` | POST | Yes | Stage 2: adaptive cleaning pipeline |
| `/train` | POST | Yes | Stage 3: model training + benchmarking |
| `/evaluate` | POST | Yes | System self-evaluation (returns JSON via fetch) |
| `/download_cleaned` | GET | Yes | Download cleaned CSV |
| `/download_report` | GET | Yes | Generate + download PDF report |
| `/static/<path>` | GET | No | Serves CSS/JS files |

**Error handlers:**
- 404 -> redirects to `/`
- 500 -> shows error message + link home

---

## 5. THE PIPELINE (4 Stages)

### Stage 0: Target Detection
- Happens via AJAX `/preview` when user selects a file
- Scores every column on: name keywords, cardinality, dtype, missing rate, distribution
- Returns ranked candidates with confidence scores
- User can override via dropdown or accept auto-detect

### Stage 1: Diagnostics + Intelligent Analysis (`/analyze`)
1. Load CSV, sanitize column names
2. Auto-sample if >10,000 rows (Render timeout protection)
3. Detect column types
4. Run 9 diagnostic checks (missing, duplicates, outliers, imbalance, leakage, noise, mixed fields, correlations, issue identification)
5. Run expert analysis (domain detection, SWOT, interventions)
6. Run intelligent engine (profile dataset -> select strategies -> build explanation panel)
7. Save all JSON artefacts
8. Render result.html

### Stage 2: Cleaning (`/clean`)
1. Load diagnostics + leakage info
2. Remove identifiers, leakage columns, duplicates
3. Drop >90% missing columns
4. Parse datetime -> year/month/day features
5. Handle mixed-type columns
6. Impute (median for numeric, mode for categorical)
7. Cap outliers (IQR)
8. Encode (OneHot for low-cardinality, Frequency for high-cardinality, Ordinal, LabelEncoder for target)
9. Create versioned backup
10. Re-diagnose cleaned data, calculate improvements

### Stage 3: Training (`/train`)
1. Detect task type (classification vs regression)
2. Split 80/20 with stratification
3. Apply SMOTE if imbalanced
4. Scale with StandardScaler
5. Train: Random Forest, Decision Tree, Logistic Regression, KNN (or user-selected)
6. Evaluate metrics, extract feature importance
7. Save best model + scaler
8. Compare original vs cleaned performance

---

## 6. KEY MODULES EXPLAINED

### `utils/target_detector.py`
- `detect_target_column(df)` -> scores all columns, returns `{recommended, confidence, reason, candidates}`
- `get_column_preview(df)` -> first 5 rows as JSON

### `utils/intelligent_engine.py`
- `profile_dataset(df, target)` -> rich JSON profile (size class, missing ratio, skewness, correlation density, cardinality, etc.)
- `select_strategies(profile)` -> 7 weighted decisions (imputation, outliers, encoding, scaling, imbalance, models, diagnostics)
- `build_explanation_panel(profile, strategies)` -> UI cards with icon, confidence %, reason
- `run_intelligent_analysis(df, target)` -> convenience wrapper returning all three

### `utils/system_evaluator.py`
- `evaluate_system(datasets)` -> runs full pipeline on each dataset, returns per-dataset metrics + aggregate stats

### `utils/report_generator.py`
- `generate_pdf_report(...)` -> professional A4 PDF using reportlab
- `generate_text_report(...)` -> legacy .txt format (fallback)

### `utils/data_analysis.py`
- `perform_diagnostics(df)` -> 9 checks, returns structured dict with identified_issues list

### `utils/data_cleaning.py`
- `clean_dataset(df, leakage_cols, target_col)` -> full cleaning pipeline, returns cleaned df

### `utils/model_training.py`
- `train_and_evaluate(df, target_col, selected_algo)` -> trains models, returns (results_dict, task_type)
- `detect_task_type(df, target_col)` -> 'classification' or 'regression'

### `utils/dataset_expert.py`
- `analyze_dataset_expertly(df, diagnostics, is_repaired)` -> SWOT, domain, quality score, interventions

### `utils/column_detector.py`
- `detect_column_types(df, target_col)` -> classifies into identifiers, datetime, numerical, nominal, ordinal

---

## 7. FRONTEND PAGES

### `splash.html`
- Dark background, animated ADIE logo, loading bar
- JavaScript auto-redirects to `/login` after 3 seconds

### `auth.html`
- Shared login/signup form (toggled by `signup` boolean)
- Dark background, white card, emerald accent bar
- Flash messages for errors

### `index.html` (Dashboard)
- Hero header with username + logout
- Drag-and-drop upload zone with live preview:
  - AJAX calls `/preview` on file select
  - Shows file name, rows x cols
  - Populates target column dropdown
  - Renders first 5 rows in scrollable table
- Default datasets dropdown
- Feature description cards

### `result.html` (Analysis Dashboard - 7 sections)
1. **Expert Report** - quality score, domain, industry, suitability badge, issues grid, interventions, SWOT
2. **Decision Transparency Panel** - 7 expandable cards showing every strategy decision with confidence bars
3. **Metadata** - rows, columns, size, export button
4. **Pipeline Control** - Execute Pipeline button or Algorithm selector + Run Benchmark
5. **Before vs After Analytics** - summary cards, 2 charts, side-by-side issues, improvement indicators
6. **ML Performance Benchmarks** - toggle original/cleaned, metric cards, comparison chart, feature importance, prediction trend, PDF download
7. **System Evaluation Dashboard** - Run Evaluation button, aggregate cards, bar chart, results table

---

## 8. DEPLOYMENT CONFIGURATION (RENDER)

### `Procfile`
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
```

### `render.yaml`
```yaml
services:
  - type: web
    name: adie-intelligence
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
    envVars:
      - key: ADIE_SECRET_KEY
        generateValue: true
      - key: OPENBLAS_NUM_THREADS
        value: "1"
      - key: OMP_NUM_THREADS
        value: "1"
      - key: MKL_NUM_THREADS
        value: "1"
```

### `requirements.txt`
```
Flask>=2.3.0
gunicorn>=21.2.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
reportlab>=4.0.0
regex>=2023.0.0
```

### Flask App Initialization (app.py lines 30-42)
```python
app = Flask(
    __name__,
    static_folder='static',
    static_url_path='/static',
    template_folder='templates'
)
app.secret_key = os.environ.get('ADIE_SECRET_KEY', 'supersecretkey')
```

### Entry Point (app.py bottom)
```python
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
```

### Critical Deployment Notes
- `static_folder='static'` and `template_folder='templates'` MUST be explicit for Render
- `$PORT` is set by Render dynamically - never hardcode it in Procfile
- `DEMO_MAX_ROWS = 10000` auto-samples large datasets to prevent timeouts
- Thread limits (`OPENBLAS_NUM_THREADS=1`) prevent memory bloat on free tier
- `uploads/` folder is ephemeral on Render (resets on redeploy) - this is fine for demo
- 404 errors redirect to `/` (prevents blank pages)

---

## 9. CSS DESIGN SYSTEM

**Color palette:**
- Primary: #065f46 (Deep Forest Green)
- Accent: #10b981 (Emerald)
- Secondary: #334155 (Charcoal)
- Danger: #991b1b (Crimson)
- Warning: #92400e (Burnt Amber)
- Background: #f8fafc (Light Grey-Blue)

**Fonts:** Plus Jakarta Sans (body), Space Grotesk (headings)

**Key CSS classes:**
- `.hero` - gradient banner
- `.card` - white card with fadeInUp animation
- `.btn-primary` / `.btn-secondary` - buttons
- `.upload-zone` - drag-drop area
- `.decision-card` - expandable details element
- `.alert` - info/success/warning messages
- `.stat-label` / `.stat-value` - metric displays
- `#loadingOverlay` - full-screen spinner
- `.badge-improved` / `.badge-declined` - green/red indicators

**All templates link CSS via:**
```html
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
```

---

## 10. AUTHENTICATION

- Users stored in `uploads/users.json` as `{"username": "password"}`
- Default: `admin` / `password123`
- Session-based: `session['user'] = username`
- `@login_required` decorator on protected routes
- Plaintext passwords (demo only, not production-safe)

---

## 11. DATA FLOW

```
User drags file into upload zone
    |
    v
/preview (AJAX) -> target_detector.py -> returns columns + preview + target
    |
    v
User confirms target, clicks "Analyse Dataset"
    |
    v
/analyze -> column_detector -> data_analysis -> dataset_expert -> intelligent_engine
    |
    v
result.html (Stage 1: Diagnostics + Decision Transparency)
    |
    v
User clicks "Execute Pipeline"
    |
    v
/clean -> data_cleaning -> version snapshot -> re-diagnose -> compare
    |
    v
result.html (Stage 2: Before/After Analytics)
    |
    v
User clicks "Run Benchmark"
    |
    v
/train -> model_training -> best_model.pkl
    |
    v
result.html (Stage 3: Performance charts + feature importance)
    |
    v
User clicks "Run System Evaluation"
    |
    v
/evaluate -> system_evaluator -> runs pipeline on all datasets
    |
    v
result.html (Evaluation Dashboard: summary + chart + table)
    |
    v
User clicks "Download PDF Report"
    |
    v
/download_report -> report_generator -> ADIE_Analysis_Report.pdf
```

---

## 12. JSON ARTEFACT STRUCTURES

### metadata.json
```json
{
  "filename": "dataset.csv",
  "size_kb": 1234.5,
  "rows": 10000,
  "columns": 15,
  "column_names": ["col1", "col2", ...],
  "types": {"col1": "int64", "col2": "object", ...},
  "target_col": "label",
  "column_types": {
    "identifiers": [],
    "datetime_cols": [],
    "numerical_cols": ["age", "income"],
    "nominal_categorical": ["gender"],
    "ordinal_categorical": ["rating"]
  }
}
```

### intelligent_analysis.json
```json
{
  "profile": {
    "n_rows": 10000, "n_cols": 15, "size_class": "medium",
    "overall_missing_ratio": 0.05, "avg_skewness": 0.8,
    "corr_density": 0.12, "target_info": {"unique_values": 2, "imbalance_ratio": 3.5}
  },
  "strategies": {
    "imputation": {"decision": "Median Imputation", "confidence": 0.87, "reason": "..."},
    "outlier_handling": {"decision": "IQR Capping", "confidence": 0.80, "reason": "..."},
    "encoding": {"decision": "Frequency + One-Hot", "confidence": 0.90, "reason": "..."},
    "scaling": {"decision": "StandardScaler", "confidence": 0.88, "reason": "..."},
    "imbalance": {"decision": "SMOTE Oversampling", "confidence": 0.80, "reason": "..."},
    "model_selection": {"decision": "Primary: Random Forest", "confidence": 0.90, "recommended_models": ["Random Forest", "Decision Tree", "Logistic Regression"]},
    "diagnostics_focus": {"decision": "Run 7 checks", "confidence": 0.95, "checks": [...]}
  },
  "explanation_panel": [
    {"category": "Imputation Strategy", "icon": "fa-fill-drip", "decision": "Median Imputation", "confidence": 0.87, "confidence_pct": 87, "reason": "..."}
  ]
}
```

### ml_results.json
```json
{
  "orig_results": {"Random Forest": {"Accuracy": 0.82, "F1-Score": 0.79, ...}},
  "cleaned_results": {"Random Forest": {"Accuracy": 0.94, "F1-Score": 0.91, ...}},
  "task_type": "classification",
  "selected_algo": "All Algorithms",
  "target_col": "label"
}
```

### eval_results.json
```json
{
  "results": [{"dataset_name": "...", "success": true, "accuracy_improvement": 0.12, ...}],
  "aggregate": {"total_datasets": 5, "success_rate": 0.80, "avg_accuracy_improvement": 0.08}
}
```

---

## 13. KNOWN LIMITATIONS

1. Passwords in plaintext (demo only)
2. Session is in-memory (lost on Render restart)
3. uploads/ is ephemeral on Render (resets on redeploy)
4. Free tier has 512MB RAM limit - large datasets may OOM
5. First request after inactivity takes ~30s (cold start)
6. No CSRF protection
7. Single-user state (all users share uploads/)
8. System evaluation limited to 5 datasets per run
9. No /predict endpoint (model trained but not exposed for inference)

---

## 14. COMMON ISSUES & FIXES

| Issue | Cause | Fix |
|---|---|---|
| 404 on all pages | app.py not at repo root | Ensure app.py is at root, not in subfolder |
| CSS not loading (black/white) | Flask can't find static/ | Explicit `static_folder='static'` in Flask() |
| Timeout on training | Dataset too large | DEMO_MAX_ROWS=10000 auto-samples |
| Session lost | Render restarted | Just log in again |
| PDF download fails | reportlab not installed | Falls back to .txt automatically |
| 502 Bad Gateway | App still starting | Wait 30s, refresh |
| Upload fails | File >50MB | Reduce file size or increase MAX_CONTENT_LENGTH |

---

## 15. HOW TO MAKE CHANGES

1. Edit files locally
2. Test: `python app.py` (runs on localhost:10000)
3. Commit + push to GitHub
4. Render auto-redeploys in ~3 minutes

**To add a new route:**
```python
@app.route('/new-page')
@login_required
def new_page():
    return render_template('new_page.html')
```

**To add a new utility module:**
1. Create `utils/new_module.py`
2. Import in app.py: `from utils.new_module import my_function`
3. Use in routes

---

*End of document. This file contains everything needed to understand, modify, debug, or extend ADIE v2.0.*
