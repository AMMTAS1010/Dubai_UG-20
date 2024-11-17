import os

import pandas as pd
from data_loader import load_preprocessed_data
from decision_tree_model import train_decision_tree, plot_decision_tree
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

def plot_decision_tree(model, feature_names, class_names, plot_file_path, fontsize=12):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, fontsize=fontsize)
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Decision Tree plot saved to {plot_file_path}")

def encode_categorical_data(X):
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if not categorical_cols.empty:
        # Apply OneHotEncoder to categorical columns
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_cats = encoder.fit_transform(X[categorical_cols])
        cat_col_names = encoder.get_feature_names_out(categorical_cols)
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=cat_col_names, index=X.index)
        
        # Drop original categorical columns and add encoded ones
        X = pd.concat([X.drop(columns=categorical_cols), encoded_cats_df], axis=1)
    
    return X

def main():
    base_dir = input("Enter the directory of preprocessed data (e.g., './Datasets/preprocessed_data/delivery'): ").strip()
    if not os.path.exists(base_dir):
        print(f"The directory {base_dir} does not exist. Please check your input.")
        return

    # Load preprocessed data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocessed_data(base_dir)

    # Ensure X_train is fully numeric
    X_train = encode_categorical_data(X_train)
    
    # Ensure y_train is a Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()

    # Train decision tree
    model = train_decision_tree(X_train, y_train, max_depth=10)
    print("Model trained successfully!")

    # Evaluate on validation set
    # Visualize the decision tree
    feature_names = list(X_train.columns)
    class_names = sorted(y_train.unique())
    plot_file_path = os.path.join(base_dir, "decision_tree_plot.png")
    plot_decision_tree(model, feature_names, class_names, plot_file_path)

if __name__ == "__main__":
    main()
