import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
from packaging import version
import sklearn
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Device selection optimized for M1 Mac GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

def load_and_preprocess_data(base_dir, dataset_type):
    """
    Load the preprocessed data from the specified directory for the given dataset type.
    """
    data_dir = os.path.join(base_dir, dataset_type)

    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    X_valid = pd.read_csv(os.path.join(data_dir, 'X_valid.csv'))
    y_valid = pd.read_csv(os.path.join(data_dir, 'y_valid.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

    # Assuming 'target' column in y files
    y_train = y_train['target']
    y_valid = y_valid['target']
    y_test = y_test['target']

    # Combine training and validation sets
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    # Identify categorical features
    categorical_features = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = [col for col in X_train_full.columns if col not in categorical_features]

    num_classes = len(y_train_full.unique())

    return X_train_full, y_train_full, X_test, y_test, num_classes, categorical_features, numerical_features

def create_logistic_regression_pipeline(categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=200, random_state=42))
    ])

    return pipeline

def evaluate_logistic_regression(model, param_grid, X_train, X_test, y_train, y_test, model_name, class_labels, results_dir):
    print(f"\nEvaluating {model_name}")

    # Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Save cross-validation results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(os.path.join(results_dir, f"{model_name}_grid_search_results.csv"), index=False)

    # Predict and evaluate on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")

    # Ensure class_labels are strings
    class_labels_str = [str(label) for label in class_labels]

    report = classification_report(y_test, y_pred, target_names=class_labels_str)
    print("Classification Report:")
    print(report)

    # Save classification report to a text file
    with open(os.path.join(results_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels_str,
        yticklabels=class_labels_str
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix.png"))
    plt.show()

    # Learning Curve
    plot_learning_curve(best_model, X_train, y_train, model_name, results_dir)

    # Feature Importance
    plot_feature_importance(best_model, X_train.columns, model_name, results_dir)

    return best_model

def plot_learning_curve(estimator, X, y, model_name, results_dir):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"{model_name}_learning_curve.png"))
    plt.show()

def plot_roc_curve(model, X_test, y_test, model_name, num_classes, results_dir):
    """
    Plot the ROC curve for multiclass classification.
    """
    # Binarize the output for multiclass
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    # Get probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        print("Model does not support probability predictions.")
        return

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 7))
    for idx, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{model_name} ROC Curve (Multiclass)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"{model_name}_ROC_curve.png"))
    plt.show()

def plot_feature_importance(model, feature_names, model_name, results_dir):
    """
    Plot feature importance for Logistic Regression.
    """
    # Get the coefficients from the logistic regression model
    coef = model.named_steps['classifier'].coef_

    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    feature_names_transformed = preprocessor.get_feature_names_out()

    # Create a DataFrame for coefficients
    coef_df = pd.DataFrame(coef.T, index=feature_names_transformed, columns=model.named_steps['classifier'].classes_)

    # Plot the coefficients
    for class_label in model.named_steps['classifier'].classes_:
        top_features = coef_df[class_label].abs().sort_values(ascending=False).head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f"Top Features for Class {class_label}")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{model_name}_Feature_Importance_Class_{class_label}.png"))
        plt.show()

# Custom MLP Classifier using PyTorch
class MLPClassifierTorch(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLPClassifierTorch, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_mlp_pytorch(X_train, y_train, X_test, y_test, num_classes, results_dir, class_labels):
    # Encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    # One-Hot Encoding for categorical features
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])

    # Standardization for numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])

    # Combine numerical and categorical features
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    # Encode target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)

    # Define model parameters
    input_size = X_train_processed.shape[1]
    hidden_sizes = [128, 64]
    model = MLPClassifierTorch(input_size, hidden_sizes, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    batch_size = 64

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        train_acc_history.append(epoch_acc)

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            total = y_test_tensor.size(0)
            correct = (predicted == y_test_tensor).sum().item()
            val_acc = correct / total
            val_acc_history.append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(results_dir, 'MLPClassifier_PyTorch.pth'))

    # Plot training and validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_acc_history, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_acc_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy for MLP Classifier')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'MLPClassifier_Accuracy.png'))
    plt.show()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y_test_encoded

    # Ensure class_labels are strings
    class_labels_str = [str(label) for label in class_labels]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels_str)
    print("Classification Report:")
    print(report)

    # Save classification report to a text file
    with open(os.path.join(results_dir, "MLPClassifier_classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels_str,
        yticklabels=class_labels_str
    )
    plt.title("MLP Classifier Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(os.path.join(results_dir, "MLPClassifier_confusion_matrix.png"))
    plt.show()