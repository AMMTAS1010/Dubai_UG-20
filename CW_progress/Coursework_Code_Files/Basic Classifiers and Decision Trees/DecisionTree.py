import os

import pandas as pd
from data_loader import load_preprocessed_data
from model_evalution import evaluate_model
from decision_tree_model import train_decision_tree, plot_decision_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# The function to train the k-Nearest Neighbors model
def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a k-NN model with the specified number of neighbors.
    Args: X_train (features), y_train (labels), n_neighbors (number of neighbors to consider).
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# The function to train the Naive Bayes model
def train_naive_bayes(X_train, y_train):
    """Trains a Gaussian Naïve Bayes classifier with given features and labels."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# A function to create a directory for saving results
def create_results_directory(base_dir, dataset_name):
    """Create results directory for the specific dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, "R4_Results", dataset_name)
    os.makedirs(results_base_dir, exist_ok=True)
    return results_base_dir

def main():
    base_dir = input("Enter the directory of preprocessed data (e.g., './Datasets/preprocessed_data/delivery'): ").strip()
    if not os.path.exists(base_dir):
        print(f"The directory {base_dir} does not exist. Please check your input.")
        return

    # Load preprocessed data
    try:
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocessed_data(base_dir)
    except FileNotFoundError as e:
        print(e)
        return

    # Extract dataset name from the input directory
    dataset_name = os.path.basename(base_dir.rstrip('/'))
    results_dir = create_results_directory(base_dir, dataset_name)

    # Letting the user choose a model
    print("\nChoose a model to train:")
    print("1. Decision Tree")
    print("2. k-Nearest Neighbors")
    print("3. Naive Bayes")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':
        # Train Decision Tree model
        model = train_decision_tree(X_train, y_train, max_depth=10)

        # Visualize the decision tree
        feature_names = list(X_train.columns)
        unique_classes = sorted(set(y_train)) # Get unique classes in the target variable
        class_names = [f'Class {cls}' for cls in unique_classes] # Create class names for the plot
        plot_file_path = os.path.join(results_dir, f"{dataset_name}_1.decision_tree_plot.png")
        plot_decision_tree(model, feature_names, class_names, plot_file_path)

        # Save Decision Tree model evaluation to a separate file
        evaluation_output_path = os.path.join(results_dir, f"{dataset_name}_1.decision_tree_model_evaluation.txt")
        evaluate_model(model, X_valid, y_valid, "Decision Tree", evaluation_output_path)
    
    elif choice == '2':
        # Train k-Nearest Neighbors model with user input for number of neighbors (k)
        n_neighbors = int(input("Enter the number of neighbors for k-NN (default: 5): ").strip() or 5)
        model = train_knn(X_train, y_train, n_neighbors)

        # Save KNN model evaluation to a separate file
        evaluation_output_path = os.path.join(results_dir, f"{dataset_name}_2.knn_model_evaluation.txt")
        evaluate_model(model, X_valid, y_valid, "k-Nearest Neighbors", evaluation_output_path)

    elif choice == '3':
        # Train Naive Bayes model
        model = train_naive_bayes(X_train, y_train)

        # Save Naive Bayes model evaluation to a separate file
        evaluation_output_path = os.path.join(results_dir, f"{dataset_name}_3.naive_bayes_model_evaluation.txt")
        evaluate_model(model, X_valid, y_valid, "Naive Bayes", evaluation_output_path)

    else:
        print("Invalid choice. Exiting...")
        return


if __name__ == "__main__":
    main()
