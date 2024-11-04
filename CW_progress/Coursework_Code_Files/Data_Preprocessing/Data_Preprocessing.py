# data_preprocessing.py

# Import Libraries
import pandas as pd
import numpy as np
import os

def load_data():
    # Load the Restaurant dataset
    restaurant_df = pd.read_csv('zomato.csv')  # Adjusted to your dataset filename

    # Load the Delivery dataset
    delivery_df = pd.read_csv('Zomato Dataset.csv')  # Adjusted to your dataset filename

    return restaurant_df, delivery_df

def preprocess_restaurant(restaurant_df):
    # Drop unnecessary columns
    if 'Unnamed: 0' in restaurant_df.columns or 'Unnamed: 0.1' in restaurant_df.columns:
        restaurant_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True, errors='ignore')

    # Strip whitespace from column names
    restaurant_df.columns = restaurant_df.columns.str.strip()

    # Print columns to verify names
    print("Columns in restaurant_df:")
    print(restaurant_df.columns.tolist())

    # Rename columns for consistency
    restaurant_df.rename(columns={
        'restaurant name': 'restaurant_name',
        'restaurant type': 'restaurant_type',
        'rate (out of 5)': 'rate_out_of_5',
        'num of ratings': 'num_of_ratings',
        'avg cost (two people)': 'avg_cost_two_people',
        'online_order': 'online_order',
        'table booking': 'table_booking',
        'cuisines type': 'cuisines_type',
        'area': 'area',
        'local address': 'local_address'
    }, inplace=True)

    # Handle missing values
    restaurant_df.dropna(inplace=True)

    # Reset index after dropping rows
    restaurant_df.reset_index(drop=True, inplace=True)

    # Convert data types if necessary
    restaurant_df['rate_out_of_5'] = pd.to_numeric(restaurant_df['rate_out_of_5'], errors='coerce')
    restaurant_df['num_of_ratings'] = pd.to_numeric(restaurant_df['num_of_ratings'], errors='coerce')
    restaurant_df['avg_cost_two_people'] = pd.to_numeric(restaurant_df['avg_cost_two_people'], errors='coerce')

    # Handle missing values after type conversion
    restaurant_df.dropna(inplace=True)
    restaurant_df.reset_index(drop=True, inplace=True)

    # Encode categorical variables
    categorical_cols = ['restaurant_name', 'restaurant_type', 'online_order', 'table_booking', 'cuisines_type', 'area']
    restaurant_df = pd.get_dummies(restaurant_df, columns=categorical_cols, drop_first=True)

    return restaurant_df

def preprocess_delivery(delivery_df):
    print("Initial delivery_df shape:", delivery_df.shape)

    # Replace various forms of 'NaN' strings with actual NaN values
    delivery_df.replace(['NaN ', 'NaN', 'nan', 'NA', 'N/A'], np.nan, inplace=True)

    # Handle missing values in critical columns
    delivery_df.dropna(subset=['Time_taken (min)', 'Delivery_person_Age', 'Delivery_person_Ratings'], inplace=True)
    delivery_df.reset_index(drop=True, inplace=True)
    print("After dropping missing critical values, delivery_df shape:", delivery_df.shape)

    # Convert data types
    delivery_df['Delivery_person_Age'] = pd.to_numeric(delivery_df['Delivery_person_Age'], errors='coerce')
    delivery_df['Delivery_person_Ratings'] = pd.to_numeric(delivery_df['Delivery_person_Ratings'], errors='coerce')
    delivery_df['multiple_deliveries'] = pd.to_numeric(delivery_df['multiple_deliveries'], errors='coerce').fillna(0)
    delivery_df['Vehicle_condition'] = pd.to_numeric(delivery_df['Vehicle_condition'], errors='coerce')
    print("After type conversions, delivery_df shape:", delivery_df.shape)

    # Handle missing values after conversions
    delivery_df.dropna(subset=['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition'], inplace=True)
    delivery_df.reset_index(drop=True, inplace=True)
    print("After dropping NaNs post conversions, delivery_df shape:", delivery_df.shape)

    # Extract numeric part from 'Time_taken (min)'
    delivery_df['Time_taken (min)'] = delivery_df['Time_taken (min)'].astype(str)
    delivery_df['Time_taken (min)'] = delivery_df['Time_taken (min)'].str.extract(r'(\d+)').astype(float)
    delivery_df.dropna(subset=['Time_taken (min)'], inplace=True)
    delivery_df.reset_index(drop=True, inplace=True)
    print("After extracting 'Time_taken (min)', delivery_df shape:", delivery_df.shape)

    # Before date and time conversions
    print("\nSample 'Order_Date' values before conversion:")
    print(delivery_df['Order_Date'].head(5))
    print("\nSample 'Time_Orderd' values before conversion:")
    print(delivery_df['Time_Orderd'].head(5))
    print("\nSample 'Time_Order_picked' values before conversion:")
    print(delivery_df['Time_Order_picked'].head(5))

    # Convert date and time columns
    delivery_df['Order_Date'] = pd.to_datetime(delivery_df['Order_Date'], errors='coerce', dayfirst=True)
    delivery_df['Time_Orderd'] = pd.to_datetime(delivery_df['Time_Orderd'], errors='coerce').dt.time
    delivery_df['Time_Order_picked'] = pd.to_datetime(delivery_df['Time_Order_picked'], errors='coerce').dt.time
    print("After converting date and time columns, delivery_df shape:", delivery_df.shape)

    # Handle missing values after date conversions
    # Optionally, you can choose not to drop rows here to retain more data
    # delivery_df.dropna(subset=['Order_Date', 'Time_Orderd', 'Time_Order_picked'], inplace=True)
    # delivery_df.reset_index(drop=True, inplace=True)
    # print("After dropping NaNs post date conversions, delivery_df shape:", delivery_df.shape)

    # Fill missing values in categorical columns with 'Unknown'
    categorical_cols_fillna = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
    for col in categorical_cols_fillna:
        delivery_df[col] = delivery_df[col].fillna('Unknown')

    # Encode categorical variables
    categorical_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
    delivery_df = pd.get_dummies(delivery_df, columns=categorical_cols, drop_first=True)
    print("After encoding categorical variables, delivery_df shape:", delivery_df.shape)

    # Create 'customer_satisfaction' based on 'Time_taken (min)'
    mean_time = delivery_df['Time_taken (min)'].mean()
    delivery_df['customer_satisfaction'] = np.where(delivery_df['Time_taken (min)'] <= mean_time, 1, 0)
    print("After creating 'customer_satisfaction', delivery_df shape:", delivery_df.shape)

    return delivery_df

def merge_datasets(restaurant_df, delivery_df):
    # For the purpose of this project, we'll proceed with the delivery_df
    # You can merge datasets if there is a common key
    data_df = delivery_df.copy()
    return data_df

def save_preprocessed_data(data_df):
    # Create a directory to save preprocessed data
    if not os.path.exists('preprocessed_data'):
        os.makedirs('preprocessed_data')
    data_df.to_csv('preprocessed_data/preprocessed_data.csv', index=False)
    print("Preprocessed data saved to 'preprocessed_data/preprocessed_data.csv'.")

def main():
    # Load data
    restaurant_df, delivery_df = load_data()

    # Preprocess datasets
    restaurant_df = preprocess_restaurant(restaurant_df)
    delivery_df = preprocess_delivery(delivery_df)

    # Merge datasets or proceed with one
    data_df = merge_datasets(restaurant_df, delivery_df)

    # Save preprocessed data
    save_preprocessed_data(data_df)

if __name__ == "__main__":
    main()