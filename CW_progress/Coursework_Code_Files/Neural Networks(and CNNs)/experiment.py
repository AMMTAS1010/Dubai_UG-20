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
    """
    Runs the CNN analysis for the Food101 dataset.
    """
    print("Running CNN model for Food101 dataset...")
    dataset_path = "../Datasets/archive/images"  
    meta_path = "../Datasets/archive/meta/meta"
    num_classes = 101  # For Food101 dataset
    dataset_name = "Food101"

    # Select CNN Model Type
    print("\nSelect CNN Model Type:")
    print("1. Pre-trained ResNet18")
    print("2. Classical CNN (from scratch)")
    model_type_choice = input("Enter the model type number (1 or 2): ")

    if model_type_choice == "1":
        model_name = "CNN_ResNet18"
        use_pretrained = True
    elif model_type_choice == "2":
        model_name = "CNN_Classical"
        use_pretrained = False
    else:
        print("Invalid model type choice. Defaulting to Pre-trained ResNet18.")
        model_name = "CNN_ResNet18"
        use_pretrained = True

    # Define training parameters
    img_height = 224  # Based on the most common image size (512x512), but resized to 224x224 for consistency
    img_width = 224
    batch_size = 64
    epochs = 100  # Increased number of epochs for deeper training

    # Create results directory
    results_dir = f"Results/{model_name}_{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    print("\nStarting CNN training...")
    print(f"Model Type: {'Pre-trained ResNet18' if use_pretrained else 'Classical CNN (from scratch)'}")
    print(f"Number of Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Results Directory: {results_dir}\n")

    # Create and train the CNN model
    model = create_cnn_model(
        dataset_path=dataset_path,
        meta_path=meta_path,
        num_classes=num_classes,
        results_dir=results_dir,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        epochs=epochs,
        use_pretrained=use_pretrained
    )

    print(f"\nTraining completed. Best model saved at '{os.path.join(results_dir, 'best_model.pth')}'")
    print(f"Training and validation accuracy plots saved at '{os.path.join(results_dir, 'cnn_training_accuracy.png')}'")
    print(f"Model checkpoints saved in '{os.path.join(results_dir, 'checkpoints')}' directory.")

    # Additional evaluation or analysis can be implemented here if needed

def run_tabular_analysis():
    """
    Runs the tabular data analysis for Delivery or Restaurant datasets.
    """
    print("Select Dataset Type:")
    print("1. Delivery Dataset")
    print("2. Restaurant Dataset")
    dataset_choice = input("Enter the dataset number (1 or 2): ")

    if dataset_choice == "1":
        dataset_type = "delivery"
        dataset_name = "Delivery"
    elif dataset_choice == "2":
        dataset_type = "restaurant"
        dataset_name = "Restaurant"
    else:
        print("Invalid dataset choice.")
        return

    base_dir = '../Datasets/preprocessed_data'  # Adjust the path accordingly

    # Load and preprocess data
    X_train_full, y_train_full, X_test, y_test, num_classes, categorical_features, numerical_features = load_and_preprocess_data(base_dir, dataset_type)

    print("\nSelect Model:")
    print("1. Logistic Regression")
    print("2. MLP Classifier")
    model_choice = input("Enter the model number (1 or 2): ")

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

        print("\nStarting Logistic Regression training and evaluation...")
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

        print("\nStarting MLP Classifier training and evaluation...")
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
    """
    Main function to select and run analysis types.
    """
    print("Select Analysis Type:")
    print("1. Image Dataset (Food101)")
    print("2. Tabular Dataset (Delivery or Restaurant)")
    analysis_choice = input("Enter the analysis number (1 or 2): ")

    if analysis_choice == "1":
        run_cnn_analysis()
    elif analysis_choice == "2":
        run_tabular_analysis()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()