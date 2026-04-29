import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
from utils.column_detector import detect_column_types

ENCODER_PATH = os.path.join('uploads', 'encoder_mappings.pkl')

def clean_dataset(df, leakage_cols=None, target_col=None, fit_encoders=True):
    """
    Idempotent data cleaning pipeline with proper type handling.
    
    Args:
        df: Input dataframe
        leakage_cols: Columns to remove (data leakage)
        target_col: Target column name (auto-detect if None)
        fit_encoders: If True, fit new encoders. If False, use saved encoders.
    """
    cleaned_df = df.copy()
    
    # FIX: Sanitize column names (remove leading/trailing whitespace)
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Auto-detect target
    if target_col is None:
        target_col = cleaned_df.columns[-1]
    
    # Also sanitize target_col if provided
    if target_col is not None:
        target_col = target_col.strip()
    
    # Verify target column exists in cleaned dataframe
    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(cleaned_df.columns)}")
    
    # Detect column types (use cleaned_df with sanitized columns)
    col_types = detect_column_types(cleaned_df, target_col)
    
    # --- BLOCK 0: REMOVE IDENTIFIER COLUMNS ---
    # Identifiers (IDs, codes, names) should be removed for ML training
    # but we keep track of them in metadata
    # CRITICAL: Never remove the target column
    if col_types['identifiers']:
        # Exclude target column from identifiers to drop
        identifiers_to_drop = [c for c in col_types['identifiers'] if c != target_col]
        if identifiers_to_drop:
            cleaned_df = cleaned_df.drop(columns=identifiers_to_drop)
            # Update column types
            for key in col_types:
                if isinstance(col_types[key], list):
                    col_types[key] = [c for c in col_types[key] if c not in identifiers_to_drop]
    
    # --- BLOCK 1: REMOVE LEAKAGE COLUMNS ---
    # CRITICAL: Never remove the target column
    if leakage_cols:
        # Exclude target column from leakage columns to drop
        cols_to_drop = [c for c in leakage_cols if c in cleaned_df.columns and c != target_col]
        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            # Update column types
            for key in col_types:
                if isinstance(col_types[key], list):
                    col_types[key] = [c for c in col_types[key] if c not in cols_to_drop]
    
    # --- BLOCK 2: REMOVE DUPLICATES ---
    cleaned_df = cleaned_df.drop_duplicates()
    
    # --- BLOCK 3: DROP HIGH-MISSING COLUMNS (>90%) ---
    # CRITICAL: Never remove the target column even if it has missing values
    missing_threshold = 0.90
    cols_to_drop = []
    for col in cleaned_df.columns:
        if col == target_col:  # Skip target column
            continue
        missing_ratio = cleaned_df[col].isnull().sum() / len(cleaned_df)
        if missing_ratio > missing_threshold:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
        # Update col_types
        for key in col_types:
            if isinstance(col_types[key], list):
                col_types[key] = [c for c in col_types[key] if c not in cols_to_drop]
    
    # --- BLOCK 4: PARSE DATETIME COLUMNS ---
    for col in col_types['datetime_cols']:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            # Extract datetime features
            cleaned_df[f'{col}_year'] = cleaned_df[col].dt.year
            cleaned_df[f'{col}_month'] = cleaned_df[col].dt.month
            cleaned_df[f'{col}_day'] = cleaned_df[col].dt.day
            cleaned_df[f'{col}_dayofweek'] = cleaned_df[col].dt.dayofweek
            # Drop original datetime column
            cleaned_df = cleaned_df.drop(columns=[col])
    
    # --- BLOCK 4.5: HANDLE MIXED FIELD INCONSISTENCIES ---
    # Convert mixed-type columns to consistent format
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            values = cleaned_df[col].astype(str)
            
            # Check if column has mixed numeric and text
            has_numeric = values.str.match(r'^-?\d*\.?\d+$').any()
            has_text = ~values.str.match(r'^-?\d*\.?\d+$').any()
            
            if has_numeric and has_text:
                # Try to convert to numeric, non-numeric becomes NaN
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                # Will be imputed in next block
    
    # --- BLOCK 5: FILL MISSING VALUES (IMPUTATION) ---
    # Use proper dtype-aware imputation
    numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from imputation
    numeric_cols = [c for c in numeric_cols if c != target_col]
    categorical_cols = [c for c in categorical_cols if c != target_col]
    
    if numeric_cols:
        imputer_num = SimpleImputer(strategy='median')
        cleaned_df[numeric_cols] = imputer_num.fit_transform(cleaned_df[numeric_cols])
    
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        cleaned_df[categorical_cols] = imputer_cat.fit_transform(cleaned_df[categorical_cols])
    
    # --- BLOCK 6: HANDLE OUTLIERS (IQR CAPPING) ---
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df[col] = np.clip(cleaned_df[col], lower_bound, upper_bound)
    
    # --- BLOCK 7: ENCODE CATEGORICAL DATA ---
    encoder_mappings = {}
    
    # Nominal: OneHotEncoder (preserves non-ordinal nature)
    # IMPROVEMENT: Handle high-cardinality columns to prevent memory errors
    nominal_cols = [c for c in col_types['nominal_categorical'] if c in cleaned_df.columns and c != target_col]
    
    # Separate high-cardinality columns (>50 unique values) for special handling
    high_cardinality_cols = []
    safe_nominal_cols = []
    
    for col in nominal_cols:
        nunique = cleaned_df[col].nunique()
        if nunique > 50:  # Too many categories - use frequency encoding instead
            high_cardinality_cols.append(col)
        else:
            safe_nominal_cols.append(col)
    
    # Process safe nominal columns with OneHotEncoder
    if safe_nominal_cols:
        if fit_encoders:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = ohe.fit_transform(cleaned_df[safe_nominal_cols].astype(str))
            encoder_mappings['nominal'] = {'encoder': ohe, 'columns': safe_nominal_cols}
        else:
            # Load saved encoder
            all_mappings = joblib.load(ENCODER_PATH)
            ohe = all_mappings['nominal']['encoder']
            encoded_data = ohe.transform(cleaned_df[safe_nominal_cols].astype(str))
        
        # Create new column names
        ohe_columns = ohe.get_feature_names_out(safe_nominal_cols)
        encoded_df = pd.DataFrame(encoded_data, columns=ohe_columns, index=cleaned_df.index)
        cleaned_df = pd.concat([cleaned_df.drop(columns=safe_nominal_cols), encoded_df], axis=1)
    
    # Handle high-cardinality columns with frequency encoding
    if high_cardinality_cols:
        encoder_mappings['high_cardinality'] = {'columns': high_cardinality_cols, 'mappings': {}}
        for col in high_cardinality_cols:
            # Frequency encoding: replace category with its frequency
            freq_map = cleaned_df[col].value_counts(normalize=True)
            encoder_mappings['high_cardinality']['mappings'][col] = freq_map.to_dict()
            cleaned_df[col] = cleaned_df[col].map(freq_map)
            cleaned_df[col] = cleaned_df[col].fillna(0)  # Handle unseen categories
    
    # Ordinal: OrdinalEncoder (preserves order)
    ordinal_cols = [c for c in col_types['ordinal_categorical'] if c in cleaned_df.columns and c != target_col]
    if ordinal_cols:
        if fit_encoders:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            cleaned_df[ordinal_cols] = oe.fit_transform(cleaned_df[ordinal_cols].astype(str))
            encoder_mappings['ordinal'] = {'encoder': oe, 'columns': ordinal_cols}
        else:
            all_mappings = joblib.load(ENCODER_PATH)
            oe = all_mappings['ordinal']['encoder']
            cleaned_df[ordinal_cols] = oe.transform(cleaned_df[ordinal_cols].astype(str))
    
    # Target encoding (LabelEncoder ONLY for target)
    # Safety check: verify target column still exists
    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' was removed during cleaning. Check if it was identified as an identifier or leakage column.")
    
    if cleaned_df[target_col].dtype == 'object' or pd.api.types.is_categorical_dtype(cleaned_df[target_col]):
        le = LabelEncoder()
        cleaned_df[target_col] = le.fit_transform(cleaned_df[target_col].astype(str))
        encoder_mappings['target'] = {'encoder': le, 'classes': le.classes_.tolist()}
    
    # Save encoder mappings
    if fit_encoders:
        # Remove sklearn objects for JSON serialization
        encoder_mappings_clean = {
            k: {'columns': v.get('columns', []), 'classes': v.get('classes', [])}
            for k, v in encoder_mappings.items()
        }
        joblib.dump(encoder_mappings, ENCODER_PATH)
        joblib.dump(encoder_mappings_clean, ENCODER_PATH.replace('.pkl', '_info.pkl'))
    
    # --- BLOCK 8: FINAL CLEANUP ---
    cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cleaned_df.fillna(0, inplace=True)
    
    return cleaned_df
