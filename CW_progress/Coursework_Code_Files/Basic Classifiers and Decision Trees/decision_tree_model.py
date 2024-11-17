from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def train_decision_tree(X_train, y_train, max_depth=None):
    """
    Trains a Decision Tree classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        max_depth (int): Maximum depth of the tree. Defaults to None.
    
    Returns:
        model: Trained Decision Tree model.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_valid, y_valid, model_name, save_path):
    """
    Evaluates a model and saves the results to a file.
    
    Args:
        model: Trained model.
        X_valid: Validation features.
        y_valid: Validation labels.
        model_name (str): Name of the model for reporting.
        save_path (str): Path to save the evaluation report.
    """
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    report = classification_report(y_valid, y_pred)
    
    print(f"Accuracy for {model_name}: {accuracy:.2f}")
    print(report)
    
    with open(save_path, 'w') as f:
        f.write(f"Accuracy for {model_name}: {accuracy:.2f}\n\n")
        f.write(report)

def plot_decision_tree(model, feature_names, class_names, save_path):
    """
    Plots and saves the Decision Tree structure.
    
    Args:
        model: Trained Decision Tree model.
        feature_names (list): List of feature names.
        class_names (list): List of class names.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.savefig(save_path)
    print(f"Decision Tree plot saved to {save_path}")
