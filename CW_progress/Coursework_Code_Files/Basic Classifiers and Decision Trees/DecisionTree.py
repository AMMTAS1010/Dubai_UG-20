# DecisionTree.py

import os
import pandas as pd
from data_loader import load_preprocessed_data
from model_evaluation import evaluate_model
from decision_tree_model import train_decision_tree, plot_decision_tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# The function to find the best k for k-Nearest Neighbors
def find_best_k_for_knn(X_train, y_train, X_valid, y_valid):
    """Find the best k for k-NN and return the best k and accuracy."""
    best_k = None
    best_accuracy = 0
    results = []

    print("\nFinding the best k for k-NN...")
    for k in range(1, 21): # Trying k values from 1 to 20
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        results.append((k, accuracy)) # Store the results for each k
        print(f"k={k}: Validation Accuracy = {accuracy:.4f}") # Print the accuracy for each k
        if accuracy > best_accuracy: # Update the best k if the current k is better
            best_k = k
            best_accuracy = accuracy

    return best_k, best_accuracy, results

# The function to train the k-Nearest Neighbors model
def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a k-NN model with the specified number of neighbors.
    Args: X_train (features), y_train (labels), n_neighbors (number of neighbors to consider).
    
    Returns:
        KNeighborsClassifier: The trained k-NN model.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# The function to train the Naive Bayes model
def train_naive_bayes(X_train, y_train):
    """
    Trains a Gaussian Naïve Bayes classifier with given features and labels.
    
    Returns:
        GaussianNB: The trained Gaussian Naïve Bayes model.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Function to plot confusion matrix for Naive Bayes
def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Plot and save the confusion matrix.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        class_names (list): List of class names.
        output_path (str): Path to save the confusion matrix plot.
    
    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title('Confusion Matrix - Naive Bayes')
    plt.savefig(output_path)
    plt.close()

# A function to create a directory for saving results
def create_results_directory(base_dir: str, dataset_name: str) -> str:
    """
    Create results directory for the specific dataset.
    
    Args:
        base_dir (str): The base directory where the results directory will be created.
        dataset_name (str): The name of the dataset for which the results directory is created.
    
    Returns:
        str: The path to the created results directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, "R4_Results", dataset_name)
    os.makedirs(results_base_dir, exist_ok=True)
    return results_base_dir

def main():
    # Get the data directory from the user
    print("Choose the dataset to load:")
    print("1. Delivery Dataset")
    print("2. Restaurant Dataset")

    choice = input("Enter your choice (1/2): ").strip()
    if choice == '1':
        base_dir = './Datasets/preprocessed_data/delivery'
    elif choice == '2':
        base_dir = './Datasets/preprocessed_data/restaurant'
    else:
        print("Invalid choice. Exiting...")
        return
    
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
        plot_file_path = os.path.join(results_dir, f"{dataset_name}_1.decision_tree_plot")
        plot_decision_tree(model, feature_names, class_names, plot_file_path, max_depth=3) # Plot the decision tree

        # Save Decision Tree model evaluation to a separate file
        evaluation_output_path = os.path.join(results_dir, f"{dataset_name}_1.decision_tree_model_evaluation.txt")
        evaluate_model(model, X_valid, y_valid, "Decision Tree", evaluation_output_path)
    
    elif choice == '2':
        # Find the best k for k-NN
        best_k, best_accuracy, results = find_best_k_for_knn(X_train, y_train, X_valid, y_valid)

        # Save best k results
        best_k_path = os.path.join(results_dir, f"{dataset_name}_2.best_k_for_knn.txt")
        with open(best_k_path, 'w') as f:
            f.write(f"Best k: {best_k}\n")
            f.write(f"Validation Accuracy: {best_accuracy:.4f}\n")
            f.write("All k results:\n")
            for k, acc in results:
                f.write(f"k={k}: Validation Accuracy = {acc:.4f}\n")

        print(f"\nBest k value saved to {best_k_path}")

        # Train k-Nearest Neighbors model with the best k (from the range 1-20)
        model = train_knn(X_train, y_train, n_neighbors=best_k)

        # Save KNN model evaluation to a separate file
        evaluation_output_path = os.path.join(results_dir, f"{dataset_name}_2.knn_model_evaluation.txt")
        evaluate_model(model, X_valid, y_valid, "k-Nearest Neighbors", evaluation_output_path)

        # Plot KNN model accuracy for different values of k
        k_values = [result[0] for result in results]
        accuracies = [result[1] for result in results]
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
        plt.title(f'Accuracy of k-NN for Different k Values (Best k={best_k})')
        plt.xlabel('k (Number of Neighbors)')
        plt.ylabel('Validation Accuracy')
        plt.grid(True)
        knn_plot_path = os.path.join(results_dir, f"{dataset_name}_2.knn_accuracy_plot.png")
        plt.savefig(knn_plot_path)
        print(f"KNN accuracy plot saved to {knn_plot_path}\n")

    elif choice == '3':
        # Train Naive Bayes model
        model = train_naive_bayes(X_train, y_train)

        # Save Naive Bayes model evaluation to a separate file
        evaluation_output_path = os.path.join(results_dir, f"{dataset_name}_3.naive_bayes_model_evaluation.txt")
        evaluate_model(model, X_valid, y_valid, "Naive Bayes", evaluation_output_path)

        # Plot confusion matrix for Naive Bayes model
        y_pred = model.predict(X_valid)  # Predict the validation set
        unique_classes = sorted(set(y_train)) # Get unique classes in the target variable
        class_names = [f'Class {cls}' for cls in unique_classes] # Create class names for the plot
        cm_plot_path = os.path.join(results_dir, f"{dataset_name}_3.naive_bayes_confusion_matrix.png")
        plot_confusion_matrix(y_valid, y_pred, class_names, cm_plot_path)

        print(f"Confusion Matrix for Naive Bayes saved to {cm_plot_path}\n")

    else:
        print("Invalid choice. Exiting...")
        return


if __name__ == "__main__":
    main()
