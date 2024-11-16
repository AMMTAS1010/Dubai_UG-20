# data_analysis_exploration.py

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_directories(dataset_type):
    """
    Create directories to save EDA results for the selected dataset type.
    """
    base_dir = f'EDA_Results/{dataset_type}'
    plots_dir = os.path.join(base_dir, 'plots')
    analysis_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    return base_dir, plots_dir, analysis_dir

def load_data(dataset_type):
    """
    Load preprocessed data based on the dataset type.
    """
    if dataset_type == "delivery":
        file_path = "/Coursework_Code_Files/Datasets/Original_Datasets/Zomato Dataset.csv"
    elif dataset_type == "restaurant":
        file_path = "/Coursework_Code_Files/Datasets/Original_Datasets/zomato.csv"
    else:
        raise ValueError("Invalid dataset type. Choose 'delivery' or 'restaurant'.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist. Please run the preprocessing script first.")
    
    data_df = pd.read_csv(file_path)
    return data_df

def exploratory_data_analysis(data_df, dataset_type, base_dir, plots_dir, analysis_dir):
    """
    Perform EDA on the given dataset based on its type.
    Save all plots and results to the specified directories.
    """
    # Display first few rows
    print(f"\nFirst 5 rows of the {dataset_type} dataset:")
    print(data_df.head())

    # Save the first rows to a text file
    with open(os.path.join(analysis_dir, "first_5_rows.txt"), "w") as f:
        f.write(data_df.head().to_string())

    # Descriptive Statistics
    print(f"\nDescriptive Statistics for {dataset_type} dataset:")
    print(data_df.describe())

    # Save descriptive statistics to a CSV file
    data_df.describe().to_csv(os.path.join(analysis_dir, "descriptive_statistics.csv"))

    # Check for missing values
    print(f"\nMissing Values in {dataset_type} dataset:")
    missing_values = data_df.isnull().sum()
    print(missing_values)

    # Save missing values information to a text file
    with open(os.path.join(analysis_dir, "missing_values.txt"), "w") as f:
        f.write(missing_values.to_string())

    # Correlation Matrix
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns for correlation in {dataset_type} dataset:")
    print(numeric_cols)

    corr_matrix = data_df[numeric_cols].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title(f'{dataset_type.capitalize()} Dataset: Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"))
    plt.close()

    if dataset_type == "delivery":
        eda_for_delivery(data_df, plots_dir)
    elif dataset_type == "restaurant":
        eda_for_restaurant(data_df, plots_dir)
    else:
        print("Invalid dataset type for specific EDA.")

def eda_for_delivery(data_df, plots_dir):
    """
    Perform EDA specific to the Delivery dataset.
    """
    print("\nPerforming EDA for Delivery dataset...")

    # Distribution of Time Taken
    plt.figure(figsize=(8, 6))
    sns.histplot(data_df['Time_taken (min)'], bins=30, kde=True)
    plt.title('Distribution of Delivery Time')
    plt.xlabel('Time Taken (min)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plots_dir, "time_taken_distribution.png"))
    plt.close()

    # Boxplot of Delivery Person Ratings vs. Time Taken
    if 'Delivery_person_Ratings' in data_df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Delivery_person_Ratings', y='Time_taken (min)', data=data_df)
        plt.title('Ratings vs. Time Taken')
        plt.xlabel('Delivery Person Ratings')
        plt.ylabel('Time Taken (min)')
        plt.savefig(os.path.join(plots_dir, "ratings_vs_time_taken.png"))
        plt.close()

def eda_for_restaurant(data_df, plots_dir):
    """
    Perform EDA specific to the Restaurant dataset.
    """
    print("\nPerforming EDA for Restaurant dataset...")

    # Distribution of Ratings
    plt.figure(figsize=(8, 6))
    sns.histplot(data_df['rate (out of 5)'], bins=30, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Ratings')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plots_dir, "ratings_distribution.png"))
    plt.close()

    # Boxplot of Average Cost vs. Ratings
    if 'avg cost (two people)' in data_df.columns and 'rate (out of 5)' in data_df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='rate (out of 5)', y='avg cost (two people)', data=data_df)
        plt.title('Average Cost vs. Ratings')
        plt.xlabel('Ratings')
        plt.ylabel('Average Cost (for Two People)')
        plt.savefig(os.path.join(plots_dir, "cost_vs_ratings.png"))
        plt.close()

    # Countplot of Online Order Preference
    if 'online_order' in data_df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='online_order', data=data_df)
        plt.title('Online Order Preference')
        plt.xlabel('Online Order')
        plt.ylabel('Count')
        plt.savefig(os.path.join(plots_dir, "online_order_preference.png"))
        plt.close()

def main():
    print("Select Dataset Type:")
    print("1. Delivery Dataset")
    print("2. Restaurant Dataset")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == "1":
        dataset_type = "delivery"
    elif choice == "2":
        dataset_type = "restaurant"
    else:
        print("Invalid choice. Please select 1 or 2.")
        return

    # Create output directories
    base_dir, plots_dir, analysis_dir = create_directories(dataset_type)

    # Load data
    data_df = load_data(dataset_type)

    # Perform EDA
    exploratory_data_analysis(data_df, dataset_type, base_dir, plots_dir, analysis_dir)

if __name__ == "__main__":
    main()