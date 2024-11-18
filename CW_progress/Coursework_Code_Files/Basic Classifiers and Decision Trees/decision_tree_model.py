# decision_tree_model

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
def plot_decision_tree(model, feature_names, class_names, save_path=None, max_depth=None):
    """
    Plots and saves the Decision Tree structure.
    
    Args:
        model: Trained Decision Tree model.
        feature_names (list): List of feature names.
        class_names (list): List of class names.
        save_path (str): Path to save the plot (without extension).
        max_depth: Maximum depth to visualize (optional)
    """
    fig, ax = plt.subplots(figsize=(26, 10))
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=class_names, 
              max_depth=max_depth, 
              filled=True, 
              fontsize=10,
              ax=ax, 
              rounded=True)
    
    plt.title("Decision Tree Visualization", fontsize=24)
    plt.subplots_adjust(left=0.025, right=0.995, top=0.94, bottom=0)  # Adjust space around the plot
    
    # Save the output plot as PNG file if save_path is provided
    if save_path:
        plt.savefig(f"{save_path}.png")
        print(f"Decision tree plot saved to {save_path}.png")
    
    plt.show()
