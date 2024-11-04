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
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Clustering](#clustering)
7. [Basic Classifiers and Decision Trees](#basic-classifiers-and-decision-trees)
8. [Neural Networks and CNNs](#neural-networks-and-cnns)
9. [Repository Contents](#repository-contents)
10. [Setup and Installation](#setup-and-installation)
11. [Additional Notes](#additional-notes)
12. [Contact Information](#contact-information)
13. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project is part of the Data Mining & Machine Learning coursework, aiming to analyze and predict customer satisfaction in the food delivery industry. By exploring and modeling the factors that influence customer satisfaction, we seek to provide actionable insights for improving service quality in food delivery operations.

## Datasets and Sources

1. **Zomato Delivery Operations Analytics Dataset**  
   - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)
   - **Description:** Contains delivery operations data.

2. **Zomato Restaurants Dataset**  
   - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/abhijitdahatonde/zomato-restaurants-dataset)
   - **Description:** Contains restaurant data on Zomato.

---

## Project Structure

```plaintext
project-root/
│
├── Main README.md
├── Weekly_labs/
├── CW_Progress/
    ├── Coursework_Code_Files/
        ├── Data_Preprocessing/
            ├── Data_Preprocessing.py
            ├── README.md  # Documentation for data preprocessing
        ├── EDA/
            ├── Exploratory_Data_Analysis.py
            ├── README.md  # Documentation for EDA
        ├── Clustering/
            ├── README.md  # Documentation for clustering analysis
        ├── Basic Classifiers and Decision Trees/
            ├── README.md  # Documentation for classifiers and decision trees
        ├── Neural Networks(and CNNs)/
            ├── README.md  # Documentation for neural networks
    ├── Datasets/
        ├── Original Datasets/
        ├── preprocessed_data/
    ├── EDA_Results/
```

## Data Preprocessing

Data preprocessing involved cleaning and preparing the datasets to ensure they are suitable for analysis and modeling. For detailed steps and explanations, refer to the [Data Preprocessing README](CW_progress/Coursework_Code_Files/Data_Preprocessing/README.md).

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to uncover patterns, detect anomalies, and test hypotheses through summary statistics and graphical representations. For detailed analysis and EDA results, refer to the [EDA README](CW_progress/Coursework_Code_Files/EDA/README.md).

## Clustering

This section contains methods and visualizations used to perform clustering on the dataset. Detailed steps and explanations are provided in the [Clustering README](CW_progress/Coursework_Code_Files/Clustering/README.md).

## Basic Classifiers and Decision Trees

This section explores various classifiers and decision trees to predict customer satisfaction. For more information, refer to the [Basic Classifiers and Decision Trees README](CW_progress/Coursework_Code_Files/Basic%20Classifiers%20and%20Decision%20Trees/README.md).

## Neural Networks and CNNs

Advanced neural network-based approaches and Convolutional Neural Networks (CNNs) are applied to predict customer satisfaction. For additional details, refer to the [Neural Networks README](CW_progress/Coursework_Code_Files/Neural%20Networks(and%20CNNs)/README.md).

---

## Repository Contents

- **Scripts**: All code scripts for data preprocessing, EDA, and modeling.
- **Data**: Raw and preprocessed datasets.
- **Images**: EDA results and any other figures.
- **Documentation**: `README.md` files for each section.

---

## Setup and Installation

To set up and run this project, follow the steps below:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/DMML-Coursework-Food-Delivery-Satisfaction.git

2. **Navigate to the Project Directory**

   ```bash
   cd DMML-Coursework-Food-Delivery-Satisfaction

3. **Install Required Dependencies**

   Ensure you have Python 3.x installed along with the necessary packages by using the following command:

   ```bash
   pip install -r requirements.txt

If `requirements.txt` is unavailable, you can install the necessary packages individually with:

   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

4. **Download the Datasets**

   Since the datasets are not included directly in the repository due to file size constraints, please download them from the links provided in the [Datasets and Sources](#datasets-and-sources) section of this README. Once downloaded, place the dataset files into the following directory:

   ```plaintext
   CW_Progress/Datasets/Original Datasets/

5. **Run Data Preprocessing**

   To clean and preprocess the data, navigate to the `Data_Preprocessing` directory and execute the preprocessing script:

   ```bash
   cd CW_Progress/Coursework_Code_Files/Data_Preprocessing
   python Data_Preprocessing.py

6. **Run Exploratory Data Analysis (EDA)**

   To perform Exploratory Data Analysis, navigate to the `EDA` directory and run the EDA script:

   ```bash
   cd CW_Progress/Coursework_Code_Files/EDA
   python Exploratory_Data_Analysis.py

7. **View Outputs**

   - **Preprocessed Data**: After executing the preprocessing script, the cleaned and processed dataset will be saved in the `CW_Progress/Datasets/preprocessed_data` directory.
   - **EDA Results**: Visualizations and analysis outcomes from the EDA are saved in the `CW_Progress/EDA_Results` directory.
   - **Logs and Output Details**: Terminal outputs during each stage provide detailed logs of the preprocessing and EDA steps, offering insights into the transformations applied and data quality checks.

---

## Additional Notes

- **Data Privacy**: Ensure compliance with all data usage and privacy policies when handling datasets.
- **Data Source Licenses**: Verify and adhere to the licenses associated with the datasets used.
- **Customization**: Feel free to adjust scripts as needed to fit specific analysis requirements or to accommodate unique dataset features.
- **Future Work**: Future directions could include further data integration, additional feature engineering, and the refinement of advanced models.

---

## Contact Information

For questions, feedback, or collaboration, please contact:

- **Abdallah Alshaqra** - [Email](mailto:ama2018@hw.ac.uk)
- **Kanishka Agarwal** - [Email](mailto:kanishka.agarwal@example.com)
- **Suhaas** - [Email](mailto:suhaas@example.com)
- **Syeda Zainab** - [Email](mailto:syeda.zainab@example.com)

---

## Acknowledgments

We extend our thanks to the contributors of the datasets and the open-source community for the tools and resources used in this project.

---

# End of Main README
