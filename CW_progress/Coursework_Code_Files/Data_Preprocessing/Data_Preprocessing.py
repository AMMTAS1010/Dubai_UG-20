import os
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to preprocess tabular data for both delivery and restaurant datasets
def preprocess_tabular_data(file_path, dataset_type, output_dir):
    """
    Preprocesses tabular data by handling missing features, balancing classes, and splitting into train/test.
    Saves the processed datasets to disk.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file {file_path} does not exist. Please check the path.")

    # Load the dataset
    data = pd.read_csv(file_path)

    # Handle missing values
    if dataset_type == "delivery":
        numerical_columns = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]
        categorical_columns = ["City", "Weather_conditions", "Festival"]
    elif dataset_type == "restaurant":
        numerical_columns = ["rate (out of 5)", "avg cost (two people)"]
        categorical_columns = ["online_order", "table booking", "restaurant type", "cuisines type", "area"]
    else:
        raise ValueError("Invalid dataset type selected.")

    # Fill missing numerical values with median
    for col in numerical_columns:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())

    # Fill missing categorical values with "Missing"
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].fillna("Missing")

    # Process dataset-specific features
    if dataset_type == "delivery":
        selected_features = [
            "Time_taken (min)",
            "Delivery_person_Age",
            # "Delivery_person_Ratings",  # Removed to prevent data leakage
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
            "Vehicle_condition",
            "multiple_deliveries",
            "Weather_conditions",
            "City",
            "Festival",
        ]
        target_column = "Delivery_person_Ratings"

        # Convert target column to categorical bins
        bins = [0, 2.5, 4.0, data[target_column].max()]
        labels = ["Low", "Medium", "High"]
        data[target_column] = pd.cut(
            data[target_column],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False
        )

        # Check for NaNs in target column after binning
        nan_count = data[target_column].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaNs found in target column after binning. Dropping these rows.")
            data = data.dropna(subset=[target_column])

        # Display class distribution
        print("Class distribution after binning:")
        print(data[target_column].value_counts())

    elif dataset_type == "restaurant":
        selected_features = [
            "num of ratings",
            "avg cost (two people)",
            "online_order",
            "table booking",
            "restaurant type",
            "cuisines type",
            "area",
        ]
        target_column = "rate (out of 5)"

        # Convert target column to categorical bins
        bins = [0, 1, 2, 3, 4, data[target_column].max()]
        labels = ["1", "2", "3", "4", "5"]
        data[target_column] = pd.cut(
            data[target_column],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False
        )

        # Check for NaNs in target column after binning
        nan_count = data[target_column].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaNs found in target column after binning. Dropping these rows.")
            data = data.dropna(subset=[target_column])

        # **Combine rare classes with too few samples**
        min_samples_required = 5  # You can adjust this threshold
        class_counts = data[target_column].value_counts()
        rare_classes = class_counts[class_counts < min_samples_required].index.tolist()

        if rare_classes:
            print(f"Combining rare classes {rare_classes} with class '3'")
            data[target_column] = data[target_column].replace(rare_classes, '3')

        # Display class distribution after addressing rare classes
        print("Class distribution after addressing rare classes:")
        print(data[target_column].value_counts())

    # Filter dataset to include only selected features and target column
    available_features = [feature for feature in selected_features if feature in data.columns]
    X = data[available_features]
    y = data[target_column]

    # Drop rows with remaining missing values in X and y
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()
    X = combined[available_features]
    y = combined[target_column]

    # Identify categorical columns in X
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    categorical_features_indices = [X.columns.get_loc(col) for col in categorical_cols]

    # Encode target variable for SMOTENC using LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Recompute class counts after addressing rare classes
    class_counts = pd.Series(y_encoded).value_counts()
    min_class_size = class_counts.min()
    print("Class counts after preprocessing:")
    print(class_counts)

    # Adjust k_neighbors based on min class size
    k_neighbors = min(min_class_size - 1, 5)
    if k_neighbors < 1:
        k_neighbors = 1  # Ensure k_neighbors is at least 1
    print(f"Using k_neighbors = {k_neighbors} for SMOTENC.")

    # Balance dataset using SMOTENC with adjusted k_neighbors
    smotenc = SMOTENC(
        categorical_features=categorical_features_indices,
        random_state=42,
        k_neighbors=k_neighbors,
    )
    X_resampled, y_resampled = smotenc.fit_resample(X, y_encoded)

    # Decode target variable back to original categories
    y_resampled = le.inverse_transform(y_resampled)

    # Check for NaNs in X_resampled and y_resampled
    if X_resampled.isna().any().any():
        print("Warning: NaNs found in X_resampled after SMOTENC.")
        X_resampled = X_resampled.dropna()
        y_resampled = y_resampled[X_resampled.index]

    if pd.isna(y_resampled).any():
        print("Warning: NaNs found in y_resampled after SMOTENC.")
        X_resampled = X_resampled[~pd.isna(y_resampled)]
        y_resampled = y_resampled[~pd.isna(y_resampled)]

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resampled, y_resampled, test_size=0.4, random_state=42, stratify=y_resampled
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed datasets to CSV files
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)

    X_valid.to_csv(os.path.join(output_dir, "X_valid.csv"), index=False)
    pd.DataFrame(y_valid, columns=["target"]).to_csv(os.path.join(output_dir, "y_valid.csv"), index=False)

    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_test, columns=["target"]).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Processed datasets saved to {output_dir}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test

# Main function to handle preprocessing logic
def main():
    print("Select Dataset Type:")
    print("1. Delivery Dataset")
    print("2. Restaurant Dataset")
    choice = input("Enter your choice (1/2): ").strip()

    # Define the base output directory
    base_output_dir = "/Coursework_Code_Files/Datasets/preprocessed_data"

    if choice == "1":
        dataset_type = "delivery"
        file_path = "/Coursework_Code_Files/Datasets/Original_Datasets/Zomato Dataset.csv"  # Adjust to match actual file path
        output_dir = os.path.join(base_output_dir, "delivery")
        preprocess_tabular_data(file_path, dataset_type, output_dir)
    elif choice == "2":
        dataset_type = "restaurant"
        file_path = "/Code/Coursework_Code_Files/Datasets/Original_Datasets/zomato.csv"  # Adjust to match actual file path
        output_dir = os.path.join(base_output_dir, "restaurant")
        preprocess_tabular_data(file_path, dataset_type, output_dir)
    else:
        print("Invalid choice. Please select 1 or 2")

if __name__ == "__main__":
    main()