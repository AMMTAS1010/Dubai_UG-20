from sklearn.metrics import classification_report, accuracy_score

# evaluates the model and prints the evaluation metrics
def evaluate_model(model, X, y, model_name, save_path=None):
    """
    Evaluates a model and saves the results to a file.
    
    Args:
        model: Trained model.
        - X: Features for evaluation
        - y: True labels
        model_name (str): Name of the model (for display purposes).
        save_path (str): Path to save the evaluation report.
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    evaluation_results = f"{model_name} Model Evaluation:\n"
    evaluation_results += f"Accuracy: {accuracy:.4f}\n\n"
    evaluation_results += f"Classification Report:\n{report}\n"
    
    print(evaluation_results)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(evaluation_results)
        print(f"{model_name} Model evaluation saved to {save_path}\n")