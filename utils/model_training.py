import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- BLOCK 1: SETUP STORAGE PATHS ---
# We define where to save the best performing model and the data scaler
# so we can use them later for predictions.
MODEL_PATH = os.path.join('uploads', 'best_model.pkl')
SCALER_PATH = os.path.join('uploads', 'scaler.pkl')

def detect_task_type(df, target_col):
    """
    Detects if we should do Classification (predicting categories) 
    or Regression (predicting numbers).
    IMPROVEMENT: Check all numeric dtypes, not just float64/int64
    """
    # Sanitize target_col
    target_col = target_col.strip()
    
    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    unique_vals = df[target_col].nunique()
    # If the target is decimal or has many unique values, it's likely a number prediction (Regression)
    if pd.api.types.is_numeric_dtype(df[target_col]) and unique_vals > 20:
        return 'regression'
    else:
        return 'classification'

def get_models(task_type):
    """
    Returns a dictionary of different ML algorithms to try out 
    based on whether we are doing classification or regression.
    """
    if task_type == 'classification':
        return {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
    else:
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }

def train_and_evaluate(df, target_col, selected_algo='All Algorithms'):
    """
    The main function that splits data, trains models, and calculates scores.
    """
    # Sanitize column names
    df = df.copy()
    df.columns = df.columns.str.strip()
    target_col = target_col.strip()
    
    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    # X is the features (data used to predict), y is the target (what we want to predict)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    task_type = detect_task_type(df, target_col)
    
    # --- BLOCK 2: DATA SPLITTING ---
    # We split the data into a Training set (80%) and a Testing set (20%).
    # The model learns from the training set and we check its accuracy on the testing set.
    stratify_y = None
    if task_type == 'classification':
        class_counts = y.value_counts()
        if (class_counts >= 2).all():
            stratify_y = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )
    
    # --- BLOCK 3: HANDLE CLASS IMBALANCE (SMOTE) ---
    # If one category has very few samples, we use SMOTE to create synthetic 
    # examples of that category so the model learns it better.
    smote_applied = False
    if task_type == 'classification':
        try:
            if (y_train.value_counts() >= 2).all() and len(y_train.unique()) > 1:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                smote_applied = True
        except Exception:
            pass

    # --- BLOCK 4: DATA SCALING ---
    # Some models work better if all numbers are in the same small range.
    # We "scale" the data and save the scaler to use it again later.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # Decide which models to run
    available_models = get_models(task_type)
    models_to_run = {}
    if selected_algo == 'All Algorithms':
        models_to_run = available_models
    elif selected_algo in available_models:
        models_to_run = {selected_algo: available_models[selected_algo]}
    else:
        models_to_run = available_models

    results = {}
    best_score = -1
    best_model = None

    # --- BLOCK 5: MODEL TRAINING & SCORING ---
    # We loop through each selected algorithm, train it, and calculate metrics.
    for name, model in models_to_run.items():
        if model is None: continue
        try:
            # Train the model
            if name in ['KNN', 'Logistic Regression', 'Linear Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            model_results = {
                'params': model.get_params(),
                'y_test': y_test.tolist()[:50],
                'y_pred': y_pred.tolist()[:50],
                'smote_applied': smote_applied
            }
            
            # Calculate metrics (Accuracy, F1 for Classification; R2, MAE for Regression)
            is_classifier = hasattr(model, 'predict_proba') or 'Classifier' in str(type(model))
            if is_classifier:
                score = accuracy_score(y_test, y_pred)
                model_results.update({
                    'Accuracy': round(score, 4),
                    'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    'F1-Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                })
            else:
                score = r2_score(y_test, y_pred)
                model_results.update({
                    'MAE': round(mean_absolute_error(y_test, y_pred), 4),
                    'MSE': round(mean_squared_error(y_test, y_pred), 4),
                    'R2 Score': round(score, 4)
                })
            
            # Keep track of the best model found so far
            if score > best_score:
                best_score = score
                best_model = model

            # Determine which features were most important for this model
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(X.columns, model.feature_importances_))
                model_results['feature_importance'] = {k: round(float(v), 4) for k, v in importances.items()}

            results[name] = model_results
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # --- BLOCK 6: SAVE BEST MODEL ---
    # We save the overall best model to a file so it can be downloaded or used later.
    if best_model:
        joblib.dump(best_model, MODEL_PATH)
            
    return results, task_type
