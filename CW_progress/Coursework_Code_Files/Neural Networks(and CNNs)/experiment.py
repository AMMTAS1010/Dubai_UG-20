from MLP_and_Linear_Classifiers import (
    load_and_preprocess_data,
    create_logistic_regression_pipeline,
    evaluate_logistic_regression,
    plot_roc_curve,
    train_mlp_pytorch,
)
from CNN_Classifier import create_cnn_model
import os
import pandas as pd

def run_cnn_analysis():
    print("Running CNN model for Food101 dataset...")
    dataset_path = "../Datasets/archive/images"  # Adjust to your actual dataset path
    meta_path = "../Datasets/archive/meta/meta"       # Adjust to your actual meta files path
    num_classes = 101  # For Food101 dataset
    model_name = "CNN_ResNet18"
    dataset_name = "Food101"

    # Create results directory
    results_dir = f"Results/{model_name}_{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Create and train the CNN model
    model = create_cnn_model(dataset_path, meta_path, num_classes, results_dir)

    # You can implement evaluation functions for the CNN model here
    # For example, plot accuracy/loss curves, save the model, etc.

def run_tabular_analysis():
    print("Select Dataset Type:")
    print("1. Delivery Dataset")
    print("2. Restaurant Dataset")
    dataset_choice = input("Enter the dataset number: ")

    if dataset_choice == "1":
        dataset_type = "delivery"
        dataset_name = "Delivery"
    elif dataset_choice == "2":
        dataset_type = "restaurant"
        dataset_name = "Restaurant"
    else:
        print("Invalid dataset choice.")
        return

    base_dir = '../Datasets/preprocessed_data'  # Adjust the path

    X_train_full, y_train_full, X_test, y_test, num_classes, categorical_features, numerical_features = load_and_preprocess_data(base_dir, dataset_type)

    print("Select Model:")
    print("1. Logistic Regression")
    print("2. MLP Classifier")
    model_choice = input("Enter the model number: ")

    # Get class labels for plotting
    class_labels = sorted(y_train_full.unique())
    class_labels_str = [str(label) for label in class_labels]  # Convert to strings

    # Create results directory
    if model_choice == "1":
        model_name = "LogisticRegression"
    elif model_choice == "2":
        model_name = "MLPClassifier_PyTorch"
    else:
        print("Invalid model choice.")
        return

    results_dir = f"Results/{model_name}_{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    if model_choice == "1":
        # Logistic Regression
        model = create_logistic_regression_pipeline(categorical_features, numerical_features)

        # Define hyperparameter grid for Logistic Regression
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'saga', 'sag'],
            'classifier__max_iter': [500, 1000]
        }

        # Evaluate Logistic Regression
        best_model = evaluate_logistic_regression(
            model,
            param_grid,
            X_train_full,
            X_test,
            y_train_full,
            y_test,
            model_name,
            class_labels_str,  # Use string labels
            results_dir
        )

        # Plot ROC Curve
        plot_roc_curve(best_model, X_test, y_test, model_name, num_classes, results_dir)

    elif model_choice == "2":
        # MLP Classifier using PyTorch

        # Train and evaluate MLP Classifier
        train_mlp_pytorch(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            num_classes,
            results_dir,
            class_labels_str  # Use string labels
        )
    else:
        print("Invalid model choice.")
        return

def main():
    print("Select Analysis Type:")
    print("1. Image Dataset (Food101)")
    print("2. Tabular Dataset (Delivery or Restaurant)")
    analysis_choice = input("Enter the analysis number: ")

    if analysis_choice == "1":
        run_cnn_analysis()
    elif analysis_choice == "2":
        run_tabular_analysis()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
        main()