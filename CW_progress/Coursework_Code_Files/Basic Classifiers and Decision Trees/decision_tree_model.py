from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Function to train the decision tree model
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

# plot the decision tree
def plot_decision_tree(model, feature_names, class_names, save_path=None):
    """
    Plots and saves the Decision Tree structure.
    
    Args:
        model: Trained Decision Tree model.
        feature_names (list): List of feature names.
        class_names (list): List of class names.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10,5))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, fontsize=8)
    plt.title("Decision Tree")
    if save_path:
        plt.savefig(save_path)
        print(f"\nDecision tree plot saved to {save_path}\n")
    plt.show()
