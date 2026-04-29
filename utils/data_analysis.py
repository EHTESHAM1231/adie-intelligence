import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.column_detector import detect_column_types

def perform_diagnostics(df):
    """
    Analyzes the dataset and returns a dictionary of diagnostics.
    This function performs a deep health check on the data to find issues.
    """
    # Initialize a dictionary to store all our findings
    diagnostics = {}
    
    # FIX: Sanitize column names
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    rows, cols = df.shape
    # We assume the last column is the target we want to predict
    target_col = df.columns[-1]
    
    # Detect column types for better analysis
    col_types = detect_column_types(df, target_col)
    
    # --- BLOCK 1: MISSING VALUES ---
    # We check each column to see if there are any empty (NaN) cells.
    # We store the total count and the count for each specific column.
    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    diagnostics['missing_values'] = {
        'total': total_missing,
        'by_column': {k: int(v) for k, v in missing_values.items()}
    }
    
    # --- BLOCK 2: DUPLICATE ROWS ---
    # We look for rows that are exactly the same. Having many duplicates
    # can make the model "memorize" certain patterns too much (overfitting).
    duplicates = int(df.duplicated().sum())
    diagnostics['duplicates'] = duplicates
    
    # --- BLOCK 3: BASIC STATISTICS ---
    # For numerical columns, we calculate the average (mean), middle point (median),
    # and how much the data varies (std). This helps understand the data range.
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        stats = num_df.describe().T[['mean', '50%', 'std']]
        stats.columns = ['mean', 'median', 'std']
        diagnostics['statistics'] = stats.to_dict(orient='index')
    else:
        diagnostics['statistics'] = {}
        
    # --- BLOCK 4: OUTLIERS (Extreme Values) ---
    # We use the Interquartile Range (IQR) method to find values that are 
    # unusually high or low compared to the rest of the data.
    outliers_count = {}
    if not num_df.empty:
        for col in num_df.columns:
            Q1 = num_df[col].quantile(0.25)
            Q3 = num_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            count = ((num_df[col] < lower_bound) | (num_df[col] > upper_bound)).sum()
            outliers_count[col] = int(count)
    
    total_outliers = int(sum(outliers_count.values()))
    diagnostics['outliers'] = {
        'total': total_outliers,
        'by_column': outliers_count
    }
    
    # --- BLOCK 5: CLASS IMBALANCE ---
    # We check if one category in the target column has way more samples 
    # than others. If it does, the model might only learn the majority category.
    class_dist = df[target_col].value_counts().to_dict()
    diagnostics['class_imbalance'] = {
        'target_column': target_col,
        'distribution': {str(k): int(v) for k, v in class_dist.items()}
    }

    # --- BLOCK 6: FEATURE CORRELATION (Data Leakage) ---
    # We check if any feature is too strongly related to the target.
    # If a correlation is near 1.0 (95%+), it might be "cheating" (data leakage).
    leakage_risk = []
    if not num_df.empty and target_col in num_df.columns:
        corr = num_df.corr()[target_col].abs().sort_values(ascending=False)
        # Exclude target itself and take top 5 most related features
        top_corr = corr[1:6].to_dict()
        diagnostics['correlations'] = {k: round(v, 4) for k, v in top_corr.items()}
        
        # If correlation is over 0.95, it's a high risk of leakage
        for col, val in top_corr.items():
            if val > 0.95:
                leakage_risk.append(col)
    else:
        diagnostics['correlations'] = {}
    diagnostics['leakage_risk'] = leakage_risk

    # --- BLOCK 7: SEMANTIC & CONSISTENCY ANALYSIS (Label Noise) ---
    # We use a K-Nearest Neighbors (KNN) model to see if a row's label 
    # matches its most similar neighbors. If not, the label might be "noisy" (wrong).
    # IMPROVEMENT: Sample large datasets to avoid memory issues
    label_noise_count = 0
    if rows > 10:
        try:
            # Sample large datasets for KNN analysis (max 10,000 rows)
            if rows > 10000:
                sample_size = min(10000, rows)
                df_knn = df.sample(n=sample_size, random_state=42)
            else:
                df_knn = df
            
            # We temporarily fill missing values and encode text to numbers for KNN
            temp_df = df_knn.copy().fillna(0)
            le = LabelEncoder()
            for col in temp_df.select_dtypes(include=['object']).columns:
                # IMPROVEMENT: Skip high-cardinality columns (>50 unique values)
                if temp_df[col].nunique() > 50:
                    temp_df = temp_df.drop(columns=[col])
                    continue
                temp_df[col] = le.fit_transform(temp_df[col].astype(str))
            
            X_diag = temp_df.drop(columns=[target_col])
            y_diag = temp_df[target_col]
            
            # Classification check: do neighbors agree with the label?
            if len(y_diag.unique()) > 1:
                scaler = StandardScaler()
                X_diag_scaled = scaler.fit_transform(X_diag)
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_diag_scaled, y_diag)
                y_pred = knn.predict(X_diag_scaled)
                # Count how many labels don't match the neighbor-based prediction
                label_noise_count = int((y_pred != y_diag).sum())
                
                # Scale back to full dataset if we sampled
                if rows > 10000:
                    noise_ratio_sample = label_noise_count / sample_size
                    label_noise_count = int(noise_ratio_sample * rows)
        except Exception:
            pass
    
    diagnostics['label_noise'] = label_noise_count

    # --- BLOCK 8: MIXED FIELD INCONSISTENCIES ---
    # Detect columns with mixed data types or inconsistent formats
    # Example: A column with both numbers and strings like "123", "N/A", "unknown"
    mixed_fields = {}
    for col in df.columns:
        if col == target_col:
            continue
        
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        # Check for mixed types in object columns
        if col_data.dtype == 'object':
            # Check if values have inconsistent formats
            values = col_data.astype(str)
            
            # Detect mixed numeric and non-numeric
            has_numeric = values.str.match(r'^-?\d*\.?\d+$').any()
            has_text = ~values.str.match(r'^-?\d*\.?\d+$').any()
            
            if has_numeric and has_text:
                # Count different types
                numeric_count = values.str.match(r'^-?\d*\.?\d+$').sum()
                text_count = (~values.str.match(r'^-?\d*\.?\d+$')).sum()
                
                mixed_fields[col] = {
                    'type': 'Mixed numeric and text',
                    'numeric_count': int(numeric_count),
                    'text_count': int(text_count),
                    'sample_values': values.head(5).tolist()
                }
            
            # Detect inconsistent date formats
            elif values.str.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any():
                # Some values look like dates, others don't
                is_date = values.str.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}')
                date_count = is_date.sum()
                non_date_count = (~is_date).sum()
                
                if date_count > 0 and non_date_count > 0:
                    mixed_fields[col] = {
                        'type': 'Mixed date formats',
                        'date_count': int(date_count),
                        'non_date_count': int(non_date_count),
                        'sample_values': values.head(5).tolist()
                    }
    
    diagnostics['mixed_fields'] = mixed_fields

    # --- BLOCK 9: ISSUE IDENTIFICATION & SEVERITY ---
    # Based on the results above, we decide if an issue is "High" or "Medium" risk.
    issues = []
    
    # Check Missing Values ratio
    mv_ratio = total_missing / (rows * cols) if (rows * cols) > 0 else 0
    if mv_ratio > 0.05:
        issues.append({'type': 'Missing Values', 'severity': 'High' if mv_ratio > 0.2 else 'Medium', 'score': mv_ratio})
        
    # Check Class Imbalance ratio
    if len(class_dist) > 1:
        counts = list(class_dist.values())
        ratio = max(counts) / min(counts) if min(counts) > 0 else 100
        if ratio > 5:
            issues.append({'type': 'Class Imbalance', 'severity': 'High' if ratio > 20 else 'Medium', 'score': ratio})
            
    # Check Redundancy (Duplicates)
    dup_ratio = duplicates / rows if rows > 0 else 0
    if dup_ratio > 0.05:
        issues.append({'type': 'Redundancy', 'severity': 'High' if dup_ratio > 0.15 else 'Medium', 'score': dup_ratio})
        
    # Check Outliers ratio
    outlier_ratio = total_outliers / (rows * len(num_df.columns)) if not num_df.empty else 0
    if outlier_ratio > 0.1:
        issues.append({'type': 'Outliers', 'severity': 'High' if outlier_ratio > 0.25 else 'Medium', 'score': outlier_ratio})

    # Check Label Noise ratio
    noise_ratio = label_noise_count / rows if rows > 0 else 0
    if noise_ratio > 0.1:
        issues.append({'type': 'Label Noise', 'severity': 'High' if noise_ratio > 0.2 else 'Medium', 'score': noise_ratio})

    # Check for Data Leakage
    if leakage_risk:
        issues.append({'type': 'Data Leakage', 'severity': 'High', 'score': len(leakage_risk)})
    
    # Check for Mixed Field Inconsistencies
    if mixed_fields:
        mixed_ratio = len(mixed_fields) / cols if cols > 0 else 0
        issues.append({'type': 'Mixed Field Inconsistencies', 'severity': 'High' if mixed_ratio > 0.2 else 'Medium', 'score': len(mixed_fields)})

    # Final list of identified issues to be shown on the dashboard
    diagnostics['identified_issues'] = issues
    
    # Store column type classification
    diagnostics['column_types'] = col_types
    
    # Add datetime column info if present
    if col_types['datetime_cols']:
        diagnostics['datetime_columns'] = col_types['datetime_cols']
    
    return diagnostics
