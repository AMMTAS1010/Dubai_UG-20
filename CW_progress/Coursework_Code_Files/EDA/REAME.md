# Exploratory Data Analysis (EDA) README

## Overview

This README provides a comprehensive guide to the Exploratory Data Analysis (EDA) process for the food delivery customer satisfaction project. The EDA process aims to identify patterns, understand data distribution, and uncover insights related to customer satisfaction, delivery times, and other influencing factors.

---

## Table of Contents

1. [Objectives](#objectives)
2. [Folder Structure](#folder-structure)
3. [Steps in EDA](#steps-in-eda)
   - [1. Descriptive Statistics](#1-descriptive-statistics)
   - [2. Missing Values Analysis](#2-missing-values-analysis)
   - [3. Correlation Analysis](#3-correlation-analysis)
   - [4. Data Visualization](#4-data-visualization)
4. [Folder Contents](#folder-contents)
5. [Running EDA](#running-eda)
6. [End of EDA README](#end-of-eda-readme)

---

## Objectives

- Summarize data attributes such as mean, median, and standard deviation.
- Visualize data to understand relationships and patterns that may influence customer satisfaction.
- Identify and address any missing data.
- Analyze the correlation between features and customer satisfaction.

---

## Folder Structure

The EDA folder contains:

- `Exploratory_Data_Analysis.py`: The main EDA script.
- `README.md`: Documentation for running the EDA and understanding each analysis step.

---

## Steps in EDA

### 1. Descriptive Statistics

- **Objective**: Summarize key statistics (e.g., mean, standard deviation) for numerical features.
- **Output**: Key metrics for features like delivery time, customer ratings, and age.
- **Code**:
    ```python
    print(data_df.describe())
    ```
- **Example**:
    - Delivery Person Age: Mean ~29.57, Std ~5.81.
    - Delivery Person Ratings: Mean ~4.63, Std ~0.33.
    - Time Taken: Mean ~26.29 minutes, Std ~9.37.

### 2. Missing Values Analysis

- **Objective**: Identify and handle missing values in the dataset.
- **Code**:
    ```python
    print(data_df.isnull().sum())
    ```
- **Notes**:
    - Identified missing values in columns like `Time_Orderd` and `Time_Order_picked`.
    - Decisions on handling missing data were based on feature importance.

### 3. Correlation Analysis

- **Objective**: Identify relationships between variables, especially with customer satisfaction.
- **Code**:
    ```python
    corr_matrix = data_df.corr()
    print(corr_matrix['customer_satisfaction'].sort_values(ascending=False))
    ```
- **Key Findings**:
    - Features with highest correlations to customer satisfaction:
      - `Time_taken (min)`: Correlation ~0.81.
      - `Delivery_person_Ratings`: Correlation ~0.31.

### 4. Data Visualization

- **Objective**: Visualize data distributions and relationships.
- **Visualizations**:
    1. **Correlation Matrix Heatmap**  
        - Visualizes correlations between all numerical features.
        - ![Correlation Matrix Heatmap](../EDA_Results/correlation_matrix.png)

    2. **Distribution of Delivery Time**  
        - Shows the distribution of delivery times.
        - ![Distribution of Delivery Time](../EDA_Results/time_taken_distribution.png)

    3. **Time Taken vs. Customer Satisfaction**  
        - Compares delivery times for satisfied and unsatisfied customers.
        - ![Time Taken vs. Customer Satisfaction](../EDA_Results/time_taken_vs_satisfaction.png)

    4. **Delivery Person Ratings vs. Time Taken**  
        - Relationship between delivery person ratings and delivery time.
        - ![Delivery Person Ratings vs. Time Taken](../EDA_Results/ratings_vs_time_taken.png)

    5. **Time Taken by Weather Conditions**  
        - Analyzes how weather affects delivery times.
        - ![Time Taken by Weather Conditions](../EDA_Results/time_taken_by_weather.png)

    6. **Time Taken by Road Traffic Density**  
        - Examines road traffic's impact on delivery time.
        - ![Time Taken by Road Traffic Density](../EDA_Results/time_taken_by_traffic.png)

    7. **Customer Satisfaction Counts**  
        - Displays count of satisfied vs. unsatisfied customers.
        - ![Customer Satisfaction Counts](../EDA_Results/customer_satisfaction_counts.png)

---

## Folder Contents

- `Exploratory_Data_Analysis.py`: Script to perform EDA and generate visualizations.
- `README.md`: Documentation for the EDA folder, including setup and explanations.

---

## Running EDA

1. **Navigate to the EDA Folder**:
    ```bash
    cd CW_Progress/Coursework_Code_Files/EDA
    ```

2. **Run the EDA Script**:
    ```bash
    python Exploratory_Data_Analysis.py
    ```

3. **View Outputs**:
    - Visualizations are saved in the `EDA_Results` folder.
    - Terminal outputs will display descriptive statistics, correlation values, and any alerts on missing data.

---

## End of EDA README
