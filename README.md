# DMML Coursework - Food Delivery Customer Satisfaction Prediction

## Project Title: Predicting Customer Satisfaction in Food Delivery Services

### Group 20

### Group Members:

- **Abdallah Alshaqra**
- **Kanishka Agarwal**
- **Suhaas**
- **Syeda Zainab**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Datasets and Sources](#datasets-and-sources)
3. [Project Structure](#project-structure)
4. [1. Data Preprocessing](#1-data-preprocessing)
   - [1.1. Data Loading](#11-data-loading)
   - [1.2. Data Cleaning](#12-data-cleaning)
   - [1.3. Feature Engineering](#13-feature-engineering)
   - [1.4. Data Integration](#14-data-integration)
   - [1.5. Data Export](#15-data-export)
5. [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [2.1. Descriptive Statistics](#21-descriptive-statistics)
   - [2.2. Missing Values Analysis](#22-missing-values-analysis)
   - [2.3. Correlation Analysis](#23-correlation-analysis)
   - [2.4. Data Visualization](#24-data-visualization)
   - [2.5. Key Findings](#25-key-findings)
6. [Repository Contents](#repository-contents)
7. [How to Run the Project](#how-to-run-the-project)
8. [Additional Notes](#additional-notes)
9. [Contact Information](#contact-information)
10. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project is part of the Data Mining & Machine Learning coursework, aiming to analyze and predict customer satisfaction in the food delivery industry. By exploring and modeling the factors that influence customer satisfaction, we seek to provide actionable insights for improving service quality in food delivery operations.

---

## Datasets and Sources

We utilized two primary datasets for this project:

1. **Zomato Delivery Operations Analytics Dataset**

   - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)
   - **Description:** Contains detailed information on Zomato's food delivery operations, including delivery personnel data, delivery times, weather conditions, road traffic density, and various operational metrics relevant to delivery performance and customer satisfaction.
   - **Dataset Filename:** `Zomato Dataset.csv`

2. **Zomato Restaurants Dataset**

   - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/abhijitdahatonde/zomato-restaurants-dataset)
   - **Description:** Provides comprehensive information about restaurants listed on Zomato, such as restaurant names, ratings, average cost for two people, cuisine types, location data, and other restaurant-specific details that may influence customer satisfaction.
   - **Dataset Filename:** `zomato.csv`

---

## Project Structure

The project is organized into the following main components:

1. [**Data Preprocessing**](#1-data-preprocessing)
2. [**Exploratory Data Analysis (EDA)**](#2-exploratory-data-analysis-eda)

---

## 1. Data Preprocessing

Data preprocessing involved cleaning and preparing the datasets to ensure they are suitable for analysis and modeling. This step is crucial to improve data quality, handle missing values, and create features that enhance model performance.

### 1.1. Data Loading

- **Objective:** Read the raw datasets into pandas DataFrames for processing.

- **Steps:**

  - Loaded the datasets using `pd.read_csv()`:

    ```python
    # Load the Restaurant dataset
    restaurant_df = pd.read_csv('zomato.csv')

    # Load the Delivery dataset
    delivery_df = pd.read_csv('Zomato Dataset.csv')
    ```

### 1.2. Data Cleaning

- **Objective:** Clean the datasets by handling missing values, correcting data types, and removing unnecessary columns.

- **Techniques and Methods Used:**

  - **Handling Missing Values:**
    - Replaced strings like `'NaN '`, `'NaN'`, `'nan'`, `'NA'`, `'N/A'` with actual `NaN` values using `replace()`.
    - Dropped rows with missing critical values using `dropna(subset=[])`.

  - **Data Type Conversion:**
    - Converted columns to appropriate data types using `pd.to_numeric()` and `pd.to_datetime()` with `errors='coerce'`.

  - **Removing Unnecessary Columns:**
    - Dropped irrelevant columns like `'Unnamed: 0'`, `'Unnamed: 0.1'` from `restaurant_df`.

  - **Standardizing Column Names:**
    - Renamed columns for consistency using `rename()`.

  - **Handling Duplicates:**
    - Checked for and removed duplicate rows using `drop_duplicates()`.

### 1.3. Feature Engineering

- **Objective:** Create new features and transform existing ones to enhance the dataset's predictive power.

- **Techniques and Methods Used:**

  - **Extracting Numerical Values from Strings:**
    - Used regular expressions with `str.extract()` to extract numerical values from strings in the `'Time_taken (min)'` column.

  - **Encoding Categorical Variables:**
    - Applied one-hot encoding to categorical features using `pd.get_dummies()`.

  - **Creating Target Variable - Customer Satisfaction:**
    - Defined `customer_satisfaction` based on whether the delivery time was less than or equal to the mean delivery time.

  - **Calculating Delivery Distance:**
    - Calculated the haversine distance between restaurant and delivery locations using latitude and longitude.

### 1.4. Data Integration

- **Objective:** Merge the delivery and restaurant datasets to create a unified dataset for analysis.

- **Steps:**

  - **Potential Integration (Future Work):**
    - Suggested enriching the delivery dataset with restaurant information based on location proximity or matching restaurant names after data cleaning.

### 1.5. Data Export

- **Objective:** Save the cleaned and processed data for further analysis.

- **Steps:**

  - Saved the preprocessed DataFrame to a CSV file using `to_csv()`.

    ```python
    delivery_df.to_csv('preprocessed_data/preprocessed_data.csv', index=False)
    ```

---

## 2. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to uncover patterns, detect anomalies, and test hypotheses through summary statistics and graphical representations.

### 2.1. Descriptive Statistics

- **Objective:** Summarize the central tendencies, dispersion, and shape of the dataset's distribution.

- **Terminal Output:**

  ```plaintext
  First 5 rows of the dataset:
         ID Delivery_person_ID  Delivery_person_Age  Delivery_person_Ratings  ...  City_Semi-Urban  City_Unknown  City_Urban  customer_satisfaction
  0  0xcdcd      DEHRES17DEL01                 36.0                      4.2  ...            False         False       False                      0
  1  0xd987      KOCRES16DEL01                 21.0                      4.7  ...            False         False       False                      1
  2  0x2784     PUNERES13DEL03                 23.0                      4.7  ...            False         False       False                      1
  3  0xc8b6     LUDHRES15DEL02                 34.0                      4.3  ...            False         False       False                      1
  4  0xdb64      KNPRES14DEL02                 24.0                      4.7  ...            False         False       False                      0

  Descriptive Statistics:
         Delivery_person_Age  Delivery_person_Ratings  Time_taken (min) 
  count         43676.000000             43676.000000           43676.000000
  mean             29.567634                 4.633774              26.288351
  std               5.814344                 0.334744               9.369864
  min              15.000000                 1.000000              10.000000
  25%              25.000000                 4.500000              19.000000
  50%              30.000000                 4.700000              26.000000
  75%              35.000000                 4.900000              32.000000
  max              50.000000                 6.000000              54.000000

### 2.2. Missing Values Analysis

- **Objective:** Identify and address missing data in the dataset.

- **Terminal Output:**

  ```plaintext
  Missing Values:
  ID                                     0
  Delivery_person_ID                     0
  Delivery_person_Age                    0
  Delivery_person_Ratings                0
  Restaurant_latitude                    0
  Restaurant_longitude                   0
  Delivery_location_latitude             0
  Delivery_location_longitude            0
  Order_Date                             0
  Time_Orderd                         4141
  Time_Order_picked                   4776
  Vehicle_condition                      0
  multiple_deliveries                    0
  Time_taken (min)                       0
  Weather_conditions_Fog                 0
  Weather_conditions_Sandstorms          0
  Weather_conditions_Stormy              0
  Weather_conditions_Sunny               0
  Weather_conditions_Unknown             0
  Weather_conditions_Windy               0
  Road_traffic_density_Jam               0
  Road_traffic_density_Low               0
  Road_traffic_density_Medium            0
  Road_traffic_density_Unknown           0
  Type_of_order_Drinks                   0
  Type_of_order_Meal                     0
  Type_of_order_Snack                    0
  Type_of_vehicle_electric_scooter       0
  Type_of_vehicle_motorcycle             0
  Type_of_vehicle_scooter                0
  Festival_Unknown                       0
  Festival_Yes                           0
  City_Semi-Urban                        0
  City_Unknown                           0
  City_Urban                             0
  customer_satisfaction                  0

### 2.3. Correlation Analysis

- **Objective:** Identify relationships between variables using the correlation matrix.

- **Terminal Output:**

  ```plaintext
  Numeric columns for correlation:
  ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
   'Restaurant_longitude', 'Delivery_location_latitude',
   'Delivery_location_longitude', 'Vehicle_condition', 'multiple_deliveries',
   'Time_taken (min)', 'customer_satisfaction']

  Top features correlated with customer satisfaction:
  Time_taken (min)               0.812620
  Delivery_person_Ratings        0.316141
  multiple_deliveries            0.284928
  Delivery_person_Age            0.258638
  Vehicle_condition              0.146278
  Delivery_location_latitude     0.014223
  Restaurant_latitude            0.012529
  Delivery_location_longitude    0.008790
  Restaurant_longitude           0.008310

- **Top Correlated Features Saved to File:** `top_features.txt`

  - **File Contents:**
    ```plaintext
    Time_taken (min)
    Delivery_person_Ratings
    multiple_deliveries
    Delivery_person_Age
    Vehicle_condition
    Delivery_location_latitude
    Restaurant_latitude
    Delivery_location_longitude
    Restaurant_longitude
    ```

  - **Correlation Matrix Heatmap:**

    ![Correlation Matrix Heatmap](images/correlation_matrix.png)

---

### 2.4. Data Visualization

1. **Correlation Matrix Heatmap**
   - **Description:** Visualizes the correlation coefficients between numerical features.
   - **Image:** ![Correlation Matrix Heatmap](CW_progress/EDA_Results/correlation_matrix.png)

2. **Distribution of Delivery Time**
   - **Description:** Shows the distribution of delivery times across all orders.
   - **Image:** ![Distribution of Delivery Time](CW_progress/EDA_Results/time_taken_distribution.png)

3. **Time Taken vs. Customer Satisfaction**
   - **Description:** Compares delivery times for satisfied and unsatisfied customers.
   - **Image:** ![Time Taken vs. Customer Satisfaction](CW_progress/EDA_Results/time_taken_vs_satisfaction.png)

4. **Delivery Person Ratings vs. Time Taken**
   - **Description:** Explores the relationship between delivery person ratings and delivery time.
   - **Image:** ![Delivery Person Ratings vs. Time Taken](CW_progress/EDA_Results/ratings_vs_time_taken.png)

5. **Time Taken by Weather Conditions**
   - **Description:** Analyzes how different weather conditions affect delivery times.
   - **Image:** ![Time Taken by Weather Conditions](CW_progress/EDA_Results/time_taken_by_weather.png)

6. **Time Taken by Road Traffic Density**
   - **Description:** Examines the impact of road traffic density on delivery times.
   - **Image:** ![Time Taken by Road Traffic Density](CW_progress/EDA_Results/time_taken_by_traffic.png)

7. **Customer Satisfaction Counts**
   - **Description:** Visualizes the count of satisfied vs. unsatisfied customers.
   - **Image:** ![Customer Satisfaction Counts](CW_progress/EDA_Results/customer_satisfaction_counts.png)

---

### 2.5. Key Findings

- **Delivery Time Impact:**
  - Longer delivery times are associated with lower customer satisfaction.
  - **Action:** Delivery services should focus on reducing delivery times to improve satisfaction.

- **Delivery Person Ratings:**
  - Higher-rated delivery personnel tend to deliver faster, leading to higher customer satisfaction.
  - **Action:** Invest in training and incentivizing delivery personnel.

- **Multiple Deliveries:**
  - Orders involving multiple deliveries tend to have longer delivery times, affecting satisfaction.
  - **Action:** Optimize delivery routing to minimize delays.

- **Weather Conditions:**
  - Adverse weather conditions like fog and storms increase delivery times.
  - **Action:** Implement strategies to mitigate weather impacts, such as adjusting estimated delivery times.

- **Road Traffic Density:**
  - High traffic density areas contribute to longer delivery times.
  - **Action:** Plan delivery routes to avoid congested areas during peak times.

---

## Repository Contents

- **Scripts:**
  - `data_preprocessing.py`: Data cleaning and preprocessing script.
  - `data_analysis_exploration.py`: EDA script with visualizations.
- **Data:**
  - `preprocessed_data/`: Directory containing the preprocessed dataset.
- **Images:**
  - `images/`: Directory for visualizations and workflow diagrams.
- **Documentation:**
  - `README.md`: Project overview and instructions.
- **Supporting Files:**
  - `top_features.txt`: List of top features correlated with customer satisfaction.

---

## How to Run the Project

To replicate the analysis:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/AMMTAS/Dubai_UG-20.git

2. **Navigate to the Project Directory:**

   ```bash
   cd DMML-Coursework-Food-Delivery-Satisfaction

3. **Install Required Dependencies:**

   Ensure you have Python 3.x installed along with the necessary packages:

   ```bash
   pip install -r requirements.txt

Alternatively, install packages individually:

   ```bash
   pip install pandas numpy matplotlib seaborn
  ```

4. **Download the Datasets:**

   Although the datasets are provided in this repository, links to the kaggle web pages for those datasets have been provided at the beginning of this readme file.

  The Dataset CSV files can be found in the Dataset folder of the project directory, inside CW_progress.

5. **Run Data Preprocessing:**

   ```bash
   python data_preprocessing.py

6. **Run Exploratory Data Analysis:**

   ```bash
   python data_analysis_exploration.py

7. **View Outputs:**

   - Preprocessed data is saved in the `preprocessed_data/` directory.
   - Visualizations are saved in the `images/` directory.
   - Terminal outputs provide detailed logs of the preprocessing and EDA steps.

---

## Additional Notes

- **Data Privacy:**
  - Ensure compliance with data usage policies and privacy regulations when handling datasets.
- **Data Sources License:**
  - Verify and adhere to the licenses associated with the datasets used.
- **Image Placeholders:**
  - Replace image placeholders with actual images generated from your analysis.
- **Customizations:**
  - Feel free to customize scripts and visualizations to suit specific analysis needs.
- **Future Work:**
  - Explore integrating the restaurant dataset more effectively.
  - Implement advanced modeling techniques in subsequent phases of the project.

---

## Contact Information

For questions or collaboration, please contact:

- **Abdallah Alshaqra** - [Email](mailto:ama2018@hw.ac.uk)
- **Kanishka Agarwal** - [Email](mailto:kanishka.agarwal@example.com)
- **Suhaas** - [Email](mailto:suhaas@example.com)
- **Syeda Zainab** - [Email](mailto:syeda.zainab@example.com)

---

## Acknowledgments

We thank the contributors of the datasets and the open-source community for the tools and resources used in this project.

---

## Conclusion

This detailed README provides an in-depth overview of the data preprocessing and exploratory data analysis steps undertaken in our project. By documenting the techniques used and the findings obtained, we have set a solid foundation for the subsequent modeling and analysis phases of our coursework project.

If you have any questions or need further assistance, please feel free to reach out to any of the team members listed above.

---

# End of README
