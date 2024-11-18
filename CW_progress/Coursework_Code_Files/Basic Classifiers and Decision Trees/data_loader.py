import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def list_csv_files(directory):
    """
    List all CSV files in the given directory.
    
    Parameters:
    - directory (str): Path to the directory.
    
    Returns:
    - List of CSV filenames.
    """
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def load_preprocessed_data(base_dir):
    """
    Loads preprocessed training, validation, and test datasets.
    
    Args:
        base_dir (str): Base directory containing the preprocessed datasets.
    
    Returns:
        - X_train (DataFrame): Training features.
        - X_valid (DataFrame): Validation features.
        - X_test (DataFrame): Test features.
        - y_train (Series): Training labels.
        - y_valid (Series): Validation labels.
        - y_test (Series): Test labels.
    """
    # List available CSV files
    available_files = list_csv_files(base_dir)
    
    required_files = ['X_train.csv', 'X_valid.csv', 'X_test.csv', 'y_train.csv', 'y_valid.csv', 'y_test.csv']
    
    # Check if all required files are present
    missing_files = [f for f in required_files if f not in available_files]
    if missing_files:
        raise FileNotFoundError(f"The following required files are missing in {base_dir}: {missing_files}")
    
    # Load preprocessed datasets
    X_train = pd.read_csv(f"{base_dir}/X_train.csv")
    y_train = pd.read_csv(f"{base_dir}/y_train.csv")["target"]
    
    X_valid = pd.read_csv(f"{base_dir}/X_valid.csv")
    y_valid = pd.read_csv(f"{base_dir}/y_valid.csv")["target"]
    
    X_test = pd.read_csv(f"{base_dir}/X_test.csv")
    y_test = pd.read_csv(f"{base_dir}/y_test.csv")["target"]
    
    # Identify categorical columns in features
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-Hot Encode categorical features
    if categorical_cols:
        # Apply OneHotEncoder to categorical columns
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_valid_cat = encoder.transform(X_valid[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])
        
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        
        # Convert to DataFrame
        X_train_cat = pd.DataFrame(X_train_cat, columns=encoded_cols, index=X_train.index)
        X_valid_cat = pd.DataFrame(X_valid_cat, columns=encoded_cols, index=X_valid.index)
        X_test_cat = pd.DataFrame(X_test_cat, columns=encoded_cols, index=X_test.index)
        
        # Drop original categorical columns and concatenate encoded columns
        X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
        X_valid = X_valid.drop(columns=categorical_cols).reset_index(drop=True)
        X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)
        
        X_train = pd.concat([X_train, X_train_cat], axis=1)
        X_valid = pd.concat([X_valid, X_valid_cat], axis=1)
        X_test = pd.concat([X_test, X_test_cat], axis=1)
    
    # Encode target variable if it's categorical
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_valid = label_encoder.transform(y_valid)
        y_test = label_encoder.transform(y_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test
