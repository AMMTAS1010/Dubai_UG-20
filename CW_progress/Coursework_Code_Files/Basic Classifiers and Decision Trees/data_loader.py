import pandas as pd

def load_preprocessed_data(base_dir):
    """
    Loads preprocessed training, validation, and test datasets.
    
    Args:
        base_dir (str): Base directory containing the preprocessed datasets.
    
    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test: Data splits.
    """
    X_train = pd.read_csv(f"{base_dir}/X_train.csv")
    y_train = pd.read_csv(f"{base_dir}/y_train.csv")["target"]
    
    X_valid = pd.read_csv(f"{base_dir}/X_valid.csv")
    y_valid = pd.read_csv(f"{base_dir}/y_valid.csv")["target"]
    
    X_test = pd.read_csv(f"{base_dir}/X_test.csv")
    y_test = pd.read_csv(f"{base_dir}/y_test.csv")["target"]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test
