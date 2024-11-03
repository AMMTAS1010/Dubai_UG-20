# data_analysis_exploration.py

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    data_df = pd.read_csv('preprocessed_data/preprocessed_data.csv')
    return data_df

def exploratory_data_analysis(data_df):
    # Display first few rows
    print("First 5 rows of the dataset:")
    print(data_df.head())

    # Descriptive Statistics
    print("\nDescriptive Statistics:")
    print(data_df.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(data_df.isnull().sum())

    # Select only numeric columns for correlation
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nNumeric columns for correlation:")
    print(numeric_cols)

    # Compute Correlation Matrix
    corr_matrix = data_df[numeric_cols].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Identify highly correlated features with the target variable
    corr_with_target = corr_matrix['customer_satisfaction'].drop('customer_satisfaction')
    top_features = corr_with_target.abs().sort_values(ascending=False).head(10).index.tolist()
    print("\nTop features correlated with customer satisfaction:")
    print(corr_with_target.abs().sort_values(ascending=False).head(10))

    # Save top features for use in modeling
    with open('top_features.txt', 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    print("\nTop features saved to 'top_features.txt'.")

    # Visualizations

    # Distribution of Time Taken
    plt.figure(figsize=(8, 6))
    sns.histplot(data_df['Time_taken (min)'], bins=30, kde=True)
    plt.title('Distribution of Delivery Time')
    plt.xlabel('Time Taken (min)')
    plt.ylabel('Frequency')
    plt.savefig('time_taken_distribution.png')
    plt.close()

    # Boxplot of Time Taken vs. Customer Satisfaction
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='customer_satisfaction', y='Time_taken (min)', data=data_df)
    plt.title('Time Taken vs. Customer Satisfaction')
    plt.xlabel('Customer Satisfaction')
    plt.ylabel('Time Taken (min)')
    plt.savefig('time_taken_vs_satisfaction.png')
    plt.close()

    # Average Time Taken by Weather Conditions
    weather_cols = [col for col in data_df.columns if 'Weather_conditions_' in col]
    weather_conditions = data_df[weather_cols + ['Time_taken (min)']]
    weather_melted = weather_conditions.melt(id_vars=['Time_taken (min)'], var_name='Weather', value_name='Value')
    weather_melted = weather_melted[weather_melted['Value'] == True]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Weather', y='Time_taken (min)', data=weather_melted)
    plt.title('Time Taken by Weather Conditions')
    plt.xlabel('Weather Conditions')
    plt.ylabel('Time Taken (min)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('time_taken_by_weather.png')
    plt.close()

    # Average Time Taken by Road Traffic Density
    traffic_cols = [col for col in data_df.columns if 'Road_traffic_density_' in col]
    traffic_conditions = data_df[traffic_cols + ['Time_taken (min)']]
    traffic_melted = traffic_conditions.melt(id_vars=['Time_taken (min)'], var_name='Traffic', value_name='Value')
    traffic_melted = traffic_melted[traffic_melted['Value'] == True]
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Traffic', y='Time_taken (min)', data=traffic_melted)
    plt.title('Time Taken by Road Traffic Density')
    plt.xlabel('Road Traffic Density')
    plt.ylabel('Time Taken (min)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('time_taken_by_traffic.png')
    plt.close()

    # Customer Satisfaction by Delivery Person Ratings
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Delivery_person_Ratings', y='Time_taken (min)', hue='customer_satisfaction', data=data_df)
    plt.title('Delivery Person Ratings vs. Time Taken')
    plt.xlabel('Delivery Person Ratings')
    plt.ylabel('Time Taken (min)')
    plt.legend(title='Customer Satisfaction')
    plt.savefig('ratings_vs_time_taken.png')
    plt.close()

    # Countplot of Customer Satisfaction
    plt.figure(figsize=(6, 4))
    sns.countplot(x='customer_satisfaction', data=data_df)
    plt.title('Customer Satisfaction Counts')
    plt.xlabel('Customer Satisfaction')
    plt.ylabel('Count')
    plt.savefig('customer_satisfaction_counts.png')
    plt.close()

def main():
    # Load data
    data_df = load_data()

    # Perform EDA
    exploratory_data_analysis(data_df)

if __name__ == "__main__":
    main()