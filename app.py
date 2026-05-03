import os

# ── Resource limits (MUST be set before importing numpy/sklearn) ──────────────
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import json
import zipfile
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
from utils.data_analysis import perform_diagnostics
from utils.adaptive_cleaning import clean_dataset, clean_dataset_adaptive, get_adaptive_diagnostics
from utils.model_training import train_and_evaluate, MODEL_PATH, SCALER_PATH
from utils.report_generator import generate_text_report, generate_pdf_report
from utils.dataset_expert import analyze_dataset_expertly
from utils.column_detector import detect_column_types
from utils.target_detector import detect_target_column, get_column_preview
from utils.intelligent_engine import run_intelligent_analysis
from utils.system_evaluator import evaluate_system
import joblib
from functools import wraps

# ── Demo mode: auto-sample large datasets to prevent Render timeouts ──────────
DEMO_MAX_ROWS = 10000

app = Flask(
    __name__,
    static_folder='static',
    static_url_path='/static',
    template_folder='templates'
)
app.secret_key = os.environ.get('ADIE_SECRET_KEY', 'supersecretkey')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
DEFAULT_DATA_FOLDER = os.path.join('data', 'default')
os.makedirs(DEFAULT_DATA_FOLDER, exist_ok=True)

# ── Authentication store ──────────────────────────────────────────────────────
USERS_FILE = os.path.join(UPLOAD_FOLDER, 'users.json')
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({"admin": "password123"}, f)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return redirect(url_for('splash'))

@app.errorhandler(500)
def server_error(e):
    return f"<h1>ADIE Error</h1><p>{str(e)}</p><a href='/'>Go Home</a>", 500


# ─────────────────────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def splash():
    try:
        return render_template('splash.html')
    except Exception:
        # Fallback if template not found (deployment issue)
        return redirect(url_for('login'))


@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('auth.html', signup=False)


@app.route('/signup')
def signup():
    return render_template('auth.html', signup=True)


@app.route('/login_post', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    if username in users and users[username] == password:
        session['user'] = username
        flash(f'Welcome back, {username}!')
        return redirect(url_for('dashboard'))
    flash('Invalid credentials. Please try again.')
    return redirect(url_for('login'))


@app.route('/signup_post', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    password = request.form.get('password')
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    if username in users:
        flash('Username already exists. Please choose another.')
        return redirect(url_for('signup'))
    users[username] = password
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
    session['user'] = username
    flash('Account created successfully! Welcome to ADIE.')
    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Successfully logged out.')
    return redirect(url_for('login'))


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    default_datasets = []
    if os.path.exists(DEFAULT_DATA_FOLDER):
        default_datasets = [f for f in os.listdir(DEFAULT_DATA_FOLDER)
                            if f.endswith(('.csv', '.zip'))]
    return render_template('index.html', default_datasets=default_datasets)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'zip'}


# ─────────────────────────────────────────────────────────────────────────────
# AJAX: column preview + target detection (called after file upload)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/preview', methods=['POST'])
@login_required
def preview():
    """
    Returns JSON with:
      - columns list
      - first 5 rows
      - recommended target column + candidates
    Used by the drag-and-drop upload zone to show a live preview.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], '_preview_' + filename)
    file.save(filepath)

    try:
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zf:
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return jsonify({"error": "No CSV inside ZIP"}), 400
                zf.extract(csv_files[0], app.config['UPLOAD_FOLDER'])
                csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_files[0])
        else:
            csv_path = filepath

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        detection = detect_target_column(df)
        preview_rows = get_column_preview(df, max_rows=5)

        return jsonify({
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "preview": preview_rows,
            "target_detection": {
                "recommended": detection["recommended"],
                "confidence": detection["confidence"],
                "reason": detection["reason"],
                "candidates": detection["candidates"][:8]   # top 8 only
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files
        for p in [filepath]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PIPELINE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage1(csv_path: str, filename: str, target_col: str):
    """
    Stage 1: load CSV, detect types, run diagnostics + expert analysis,
    run intelligent engine, save all JSON artefacts.
    Returns (df, metadata, diagnostics, expert_report, intelligent_analysis).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Demo mode: sample large datasets to prevent Render timeouts
    if len(df) > DEMO_MAX_ROWS:
        df = df.sample(n=DEMO_MAX_ROWS, random_state=42).reset_index(drop=True)
        df.to_csv(csv_path, index=False)  # overwrite with sampled version

    # Validate target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. "
                         f"Available: {df.columns.tolist()}")

    col_types = detect_column_types(df, target_col)

    metadata = {
        "filename": filename,
        "size_kb": round(os.path.getsize(csv_path) / 1024, 2),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "types": df.dtypes.astype(str).to_dict(),
        "target_col": target_col,
        "column_types": {
            "identifiers":        col_types['identifiers'],
            "datetime_cols":      col_types['datetime_cols'],
            "numerical_cols":     col_types['numerical_cols'],
            "nominal_categorical": col_types['nominal_categorical'],
            "ordinal_categorical": col_types['ordinal_categorical']
        }
    }

    diagnostics = perform_diagnostics(df)
    # Attach target col to diagnostics so downstream stages can read it
    diagnostics['target_col'] = target_col

    expert_report = analyze_dataset_expertly(df, diagnostics)

    # Intelligent engine analysis
    intelligent_analysis = run_intelligent_analysis(df, target_col)

    # Persist artefacts
    _save_json('metadata.json', metadata)
    _save_json('diagnostics.json', diagnostics)
    _save_json('expert_report.json', expert_report)
    _save_json('intelligent_analysis.json', intelligent_analysis)

    return df, metadata, diagnostics, expert_report, intelligent_analysis


def _save_json(filename: str, data):
    path = os.path.join(UPLOAD_FOLDER, filename)
    with open(path, 'w') as f:
        json.dump(data, f, default=str)


def _load_json(filename: str):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE (file upload)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Please upload a valid CSV or ZIP file')
        return redirect(url_for('dashboard'))

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract CSV from ZIP if needed
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                flash('No CSV file found inside the ZIP')
                return redirect(url_for('dashboard'))
            zip_ref.extract(csv_files[0], app.config['UPLOAD_FOLDER'])
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_files[0])
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(csv_path, target_path)
    else:
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(filepath, target_path)

    # Determine target column
    # User may have submitted a selection from the preview dropdown
    user_target = request.form.get('target_column', '').strip()
    if not user_target or user_target == '__auto__':
        df_tmp = pd.read_csv(target_path)
        df_tmp.columns = df_tmp.columns.str.strip()
        detection = detect_target_column(df_tmp)
        target_col = detection['recommended']
    else:
        target_col = user_target

    try:
        df, metadata, diagnostics, expert_report, intelligent_analysis = \
            _run_stage1(target_path, filename, target_col)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for('dashboard'))

    return render_template(
        'result.html',
        diagnostics=diagnostics,
        expert_report=expert_report,
        metadata=metadata,
        filename=filename,
        intelligent_analysis=intelligent_analysis
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE DEFAULT DATASET
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/analyze_default', methods=['POST'])
@login_required
def analyze_default():
    selected_file = request.form.get('default_file')
    if not selected_file:
        flash('No default file selected')
        return redirect(url_for('dashboard'))

    source_path = os.path.join(DEFAULT_DATA_FOLDER, selected_file)
    if not os.path.exists(source_path):
        flash('Selected default file not found')
        return redirect(url_for('dashboard'))

    target_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')

    if selected_file.endswith('.zip'):
        with zipfile.ZipFile(source_path, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                flash('No CSV file found inside the ZIP')
                return redirect(url_for('dashboard'))
            zip_ref.extract(csv_files[0], app.config['UPLOAD_FOLDER'])
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_files[0])
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)
    else:
        shutil.copy(source_path, target_path)

    # Auto-detect target
    df_tmp = pd.read_csv(target_path)
    df_tmp.columns = df_tmp.columns.str.strip()
    detection = detect_target_column(df_tmp)
    target_col = detection['recommended']

    try:
        df, metadata, diagnostics, expert_report, intelligent_analysis = \
            _run_stage1(target_path, selected_file, target_col)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for('dashboard'))

    return render_template(
        'result.html',
        diagnostics=diagnostics,
        expert_report=expert_report,
        metadata=metadata,
        filename=selected_file,
        intelligent_analysis=intelligent_analysis
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/clean', methods=['POST'])
@login_required
def clean():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
    if not os.path.exists(filepath):
        flash('No dataset found to clean')
        return redirect(url_for('dashboard'))

    # Load persisted artefacts
    old_diag = _load_json('diagnostics.json') or {}
    metadata = _load_json('metadata.json') or {}

    # Resolve target column (from metadata → form → auto-detect)
    target_col = (
        request.form.get('target_column', '').strip()
        or metadata.get('target_col', '')
        or old_diag.get('target_col', '')
    )

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if not target_col or target_col not in df.columns:
        detection = detect_target_column(df)
        target_col = detection['recommended']

    leakage_cols = old_diag.get('leakage_risk', [])
    leakage_cols = [c for c in leakage_cols if c != target_col]

    # Diagnostics BEFORE cleaning
    orig_diagnostics = perform_diagnostics(df)
    orig_diagnostics['target_col'] = target_col

    # Run intelligent engine BEFORE cleaning to generate strategy hints
    intelligent_analysis_pre = run_intelligent_analysis(df, target_col)

    # Versioned backup
    version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'version_{version_timestamp}')
    os.makedirs(version_dir, exist_ok=True)
    shutil.copy(filepath, os.path.join(version_dir, 'before_clean.csv'))

    # Clean using Adaptive Data Preparation Engine
    # Strategy hints from intelligent engine guide the adaptive decisions
    cleaned_df, adaptive_report = clean_dataset_adaptive(
        df, target_col, leakage_cols=leakage_cols, verbose=False,
        strategy_hints=intelligent_analysis_pre
    )

    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_dataset.csv')
    cleaned_df.to_csv(cleaned_filepath, index=False)
    shutil.copy(cleaned_filepath, os.path.join(version_dir, 'after_clean.csv'))

    # Diagnostics AFTER cleaning (use adaptive diagnostics for richer info)
    diagnostics = perform_diagnostics(cleaned_df)
    diagnostics['target_col'] = target_col

    expert_report = analyze_dataset_expertly(cleaned_df, diagnostics, is_repaired=True)
    intelligent_analysis = run_intelligent_analysis(cleaned_df, target_col)

    # Version info
    orig_issue_types = {i['type'] for i in orig_diagnostics.get('identified_issues', [])}
    clean_issue_types = {i['type'] for i in diagnostics.get('identified_issues', [])}
    resolved = orig_issue_types - clean_issue_types

    version_info = {
        'timestamp': version_timestamp,
        'original_rows': len(df),
        'cleaned_rows': len(cleaned_df),
        'original_columns': len(df.columns),
        'cleaned_columns': len(cleaned_df.columns),
        'original_issues': len(orig_diagnostics.get('identified_issues', [])),
        'cleaned_issues': len(diagnostics.get('identified_issues', [])),
        'improvements': {
            'issues_resolved': len(resolved),
            'resolved_list': list(resolved),
            'rows_removed': len(df) - len(cleaned_df),
            'columns_removed': 0,  # GUARANTEED: no columns dropped
            'columns_added': adaptive_report.get('dataset_info', {}).get('columns_added', 0),
        },
        'adaptive_engine': {
            'dataset_type': adaptive_report.get('dataset_type', {}).get('primary_type', 'unknown'),
            'domain': adaptive_report.get('domain', {}).get('detected_domain', 'unknown'),
            'transformations': adaptive_report.get('transformations', {}).get('total', 0),
            'columns_dropped': 0,
        }
    }
    with open(os.path.join(version_dir, 'version_info.json'), 'w') as f:
        json.dump(version_info, f)

    # Update metadata with new column info
    if metadata:
        metadata['target_col'] = target_col
        if 'column_types' in diagnostics:
            metadata['column_types'] = diagnostics['column_types']

    # Persist
    _save_json('original_diagnostics.json', orig_diagnostics)
    _save_json('diagnostics.json', diagnostics)
    _save_json('expert_report.json', expert_report)
    _save_json('intelligent_analysis.json', intelligent_analysis)
    _save_json('metadata.json', metadata)
    _save_json('adaptive_report.json', adaptive_report)

    flash('ADIE Pipeline: Dataset successfully repaired and optimized!')

    return render_template(
        'result.html',
        diagnostics=diagnostics,
        expert_report=expert_report,
        metadata=metadata,
        filename=metadata.get('filename', 'dataset.csv'),
        cleaned=True,
        orig_diagnostics=orig_diagnostics,
        intelligent_analysis=intelligent_analysis,
        adaptive_report=adaptive_report
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/train', methods=['POST'])
@login_required
def train():
    orig_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_dataset.csv')
    selected_algo = request.form.get('algorithm', 'All Algorithms')

    if not os.path.exists(orig_filepath):
        flash('Original dataset not found')
        return redirect(url_for('dashboard'))

    # Load persisted artefacts
    metadata = _load_json('metadata.json') or {}
    expert_report = _load_json('expert_report.json')
    diagnostics = _load_json('diagnostics.json') or {}
    intelligent_analysis = _load_json('intelligent_analysis.json')

    # Resolve target column
    target_col = (
        request.form.get('target_column', '').strip()
        or metadata.get('target_col', '')
        or diagnostics.get('target_col', '')
    )

    df_orig = pd.read_csv(orig_filepath)
    df_orig.columns = df_orig.columns.str.strip()

    if not target_col or target_col not in df_orig.columns:
        detection = detect_target_column(df_orig)
        target_col = detection['recommended']

    # Process original for baseline training
    df_orig_processed = clean_dataset(df_orig, target_col=target_col)

    orig_results, task_type = train_and_evaluate(
        df_orig_processed, target_col, selected_algo
    )

    cleaned_results = None
    if os.path.exists(cleaned_filepath):
        df_cleaned = pd.read_csv(cleaned_filepath)
        df_cleaned.columns = df_cleaned.columns.str.strip()
        if target_col in df_cleaned.columns:
            cleaned_results, _ = train_and_evaluate(df_cleaned, target_col, selected_algo)

    # Persist ML results
    results_data = {
        'orig_results': orig_results,
        'cleaned_results': cleaned_results,
        'task_type': task_type,
        'selected_algo': selected_algo,
        'target_col': target_col
    }
    _save_json('ml_results.json', results_data)

    return render_template(
        'result.html',
        orig_results=orig_results,
        cleaned_results=cleaned_results,
        task_type=task_type,
        selected_algo=selected_algo,
        expert_report=expert_report,
        metadata=metadata,
        diagnostics=diagnostics,
        filename=metadata.get('filename', 'dataset.csv'),
        trained=True,
        intelligent_analysis=intelligent_analysis
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/evaluate', methods=['POST'])
@login_required
def evaluate():
    """
    Run ADIE self-evaluation across all available datasets in uploads/ and
    data/default/.  Returns JSON (called via fetch from the UI).
    """
    datasets = []

    # Collect datasets from uploads folder
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.endswith('.csv') and not fname.startswith('_'):
            # Skip internal artefact files
            skip_names = {
                'current_dataset.csv', 'cleaned_dataset.csv',
                'analysis_report.txt'
            }
            if fname in skip_names:
                continue
            datasets.append({
                "name": fname,
                "path": os.path.join(UPLOAD_FOLDER, fname),
                "target": None
            })

    # Collect from version folders (use after_clean.csv)
    for entry in os.scandir(UPLOAD_FOLDER):
        if entry.is_dir() and entry.name.startswith('version_'):
            after = os.path.join(entry.path, 'after_clean.csv')
            if os.path.exists(after):
                datasets.append({
                    "name": f"{entry.name}/after_clean",
                    "path": after,
                    "target": None
                })

    # Collect default datasets
    if os.path.exists(DEFAULT_DATA_FOLDER):
        for fname in os.listdir(DEFAULT_DATA_FOLDER):
            if fname.endswith('.csv'):
                datasets.append({
                    "name": f"[default] {fname}",
                    "path": os.path.join(DEFAULT_DATA_FOLDER, fname),
                    "target": None
                })

    if not datasets:
        return jsonify({"error": "No datasets found for evaluation"}), 400

    # Limit to 5 datasets to keep response time reasonable
    datasets = datasets[:5]

    eval_results = evaluate_system(datasets)

    # Persist for PDF report
    _save_json('eval_results.json', eval_results)

    return jsonify(eval_results)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/download_cleaned')
@login_required
def download_cleaned():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_dataset.csv')
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name='cleaned_dataset.csv')
    flash('Cleaned dataset not found')
    return redirect(url_for('dashboard'))


@app.route('/download_report')
@login_required
def download_report():
    """Generate and serve a PDF report (falls back to TXT if reportlab missing)."""
    diag_path = os.path.join(UPLOAD_FOLDER, 'diagnostics.json')
    res_path = os.path.join(UPLOAD_FOLDER, 'ml_results.json')
    expert_path = os.path.join(UPLOAD_FOLDER, 'expert_report.json')

    if not (os.path.exists(diag_path) and os.path.exists(expert_path)):
        flash('Please perform analysis before downloading the report.')
        return redirect(url_for('dashboard'))

    diagnostics = _load_json('diagnostics.json') or {}
    expert_report = _load_json('expert_report.json') or {}
    eval_results = _load_json('eval_results.json')

    orig_results = {}
    cleaned_results = {}
    task_type = 'classification'
    selected_algo = 'All Algorithms'

    if os.path.exists(res_path):
        results_data = _load_json('ml_results.json') or {}
        orig_results = results_data.get('orig_results', {})
        cleaned_results = results_data.get('cleaned_results', {})
        task_type = results_data.get('task_type', 'classification')
        selected_algo = results_data.get('selected_algo', 'All Algorithms')

    pdf_path = os.path.join(UPLOAD_FOLDER, 'analysis_report.pdf')

    actual_path = generate_pdf_report(
        diagnostics=diagnostics,
        expert_report=expert_report,
        orig_results=orig_results,
        cleaned_results=cleaned_results,
        task_type=task_type,
        selected_algo=selected_algo,
        output_path=pdf_path,
        eval_results=eval_results
    )

    ext = os.path.splitext(actual_path)[1]
    download_name = f'ADIE_Analysis_Report{ext}'
    return send_file(actual_path, as_attachment=True, download_name=download_name)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
