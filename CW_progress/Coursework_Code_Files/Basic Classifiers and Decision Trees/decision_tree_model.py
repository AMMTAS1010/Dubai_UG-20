# decision_tree_model

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source # Source class allows to render the graph in Jupyter notebook

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
    # Export the decision tree as a dot file
    dot_data = export_graphviz( # export_graphviz is used to export the decision tree as a dot file
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, # Fill the boxes with colors
        rounded=True, # Rounded corners of the boxes
        special_characters=True,
        max_depth=max_depth  # Limit the depth if specified
    )
    graph = Source(dot_data) # Load the dot data into a graph
        
    if save_path:
        # Save as PNG file
        graph.render(filename=save_path, format='png', cleanup=True) # render the graph as a PNG file
        print(f"Decision tree plot saved to {save_path}.png")
    
    graph.view()
