# Data Preprocessing

## Overview

This folder contains scripts and information related to the data preprocessing stage for our project on customer satisfaction in food delivery. Data preprocessing is crucial for ensuring that the datasets are clean, consistent, and ready for analysis. This stage involves handling missing values, transforming data types, creating new features, and standardizing data formats across datasets.

## Preprocessing Steps

### 1. Data Loading

- **Objective**: Load raw datasets from the Datasets directory.
- **Datasets Used**:
  - `Zomato Dataset.csv`: Delivery operations data.
  - `zomato.csv`: Restaurant information data.

- **Code**:
  ```python
  # Load the Restaurant dataset
  restaurant_df = pd.read_csv('../Datasets/zomato.csv')

  # Load the Delivery dataset
  delivery_df = pd.read_csv('../Datasets/Zomato Dataset.csv')

### 2. Data Cleaning

- **Objective**: Handle missing values, correct data types, and drop unnecessary columns.

- **Techniques**:
  - **Missing Values**: Replace placeholders (`NaN`, `nan`, etc.) with actual `NaN` values.
  - **Data Type Conversion**: Convert columns to appropriate data types for consistency.
  - **Removing Unnecessary Columns**: Drop columns that do not contribute to the analysis.
  - **Standardize Column Names**: Rename columns for consistency across datasets.

- **Code**:
  ```python
  # Replace placeholders with NaNs
  delivery_df.replace(['NaN ', 'NaN', 'nan', 'NA', 'N/A'], np.nan, inplace=True)

  # Convert types
  delivery_df['Delivery_person_Age'] = pd.to_numeric(delivery_df['Delivery_person_Age'], errors='coerce')

  # Drop unnecessary columns
  restaurant_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True, errors='ignore')

  # Standardize column names
  restaurant_df.rename(columns={
      'restaurant name': 'restaurant_name',
      'rate (out of 5)': 'rate_out_of_5',
  }, inplace=True)

  ### 3. Feature Engineering

- **Objective**: Create new features and transform existing ones to enhance the dataset’s predictive power.

- **Techniques**:
  - **Extract Numerical Values**: Extract numeric information from text fields.
  - **Encoding Categorical Variables**: One-hot encode categorical features for analysis.
  - **Create Target Variable**: Define `customer_satisfaction` based on delivery time.

- **Code**:
  ```python
  # Extract numerical values from 'Time_taken (min)' column
  delivery_df['Time_taken (min)'] = delivery_df['Time_taken (min)'].str.extract(r'(\d+)').astype(float)

  # One-hot encode categorical features
  categorical_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order']
  delivery_df = pd.get_dummies(delivery_df, columns=categorical_cols, drop_first=True)

  # Define customer satisfaction based on average delivery time
  mean_time = delivery_df['Time_taken (min)'].mean()
  delivery_df['customer_satisfaction'] = np.where(delivery_df['Time_taken (min)'] <= mean_time, 1, 0)

### 4. Data Integration

- **Objective**: Integrate relevant data from both datasets for enhanced analysis.
- **Note**: The datasets were not merged initially due to the lack of a common key. Future improvements could involve enriching the delivery data with additional restaurant information.

---

### 5. Data Export

- **Objective**: Save the cleaned and processed dataset to the `preprocessed_data` folder for further analysis.

- **Code**:
  ```python
  delivery_df.to_csv('../preprocessed_data/preprocessed_data.csv', index=False)

  ### Folder Contents

- **Data_Preprocessing.py**: Script for executing all preprocessing steps.
- **README.md**: Documentation for data preprocessing.

---

### Running Data Preprocessing

1. Ensure that all necessary dependencies are installed.
2. Navigate to the `Data_Preprocessing` folder.
3. Run the following command:

   ```bash
   python Data_Preprocessing.py

# End of Data_Preprocessing README

