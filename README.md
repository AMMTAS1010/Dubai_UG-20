# DMML Coursework - Food Delivery Customer Satisfaction Prediction

## Project Title: Predicting Customer Satisfaction in Food Delivery Services

**Group 20**

**Group Members:**

- Abdallah Alshaqra
- Kanishka Agarwal
- Suhaas
- Syeda Zainab

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
11. [Key Insights](#key-insights)
12. [Future Work](#future-work)
13. [Additional Notes](#additional-notes)
14. [Contact Information](#contact-information)
15. [Acknowledgments](#acknowledgments)

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
3. **Food101 Image Dataset**
   - **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/kmader/food41/data?select=images)
   - **Description:** contains 101 classes of international dishes.


## Project Structure

- **project-root/**
  - **Main README.md**
  - **Weekly_labs/**
  - **CW_Progress/**
    - **Coursework_Code_Files/**
      - **Data_Preprocessing/**
        - `Data_Preprocessing.py`
        - `Data_preprocessing.ipynb`
        - `README.md`  <!-- Documentation for data preprocessing -->
      - **EDA/**
        - `Exploratory_Data_Analysis.py`
        - `EDA.ipynb`
        - **EDA_Results/**
        - `README.md`  <!-- Documentation for EDA -->
      - **Clustering/**
        - **Results/**
        - `clustering_algorithms.py`
        - `clustering_utils.py`
        - `run_clustering.py`
        - `clustering.ipynb`
        - `README.md`  <!-- Documentation for clustering analysis -->
      - **Basic Classifiers and Decision Trees/**
        - **R4_Results/**
        - `BasicClassifiers.ipynb`
        - `data_loader.py`
        - `decision_tree_model.py`
        - `DecisionTree.py`
        - `model_evaluation.py`
        - `README.md`  <!-- Documentation for classifiers and decision trees -->
      - **Neural Networks (and CNNs)/**
        - **Results/**
        - `CNN_Classifier.py`
        - `experiment.py`
        - `Find_Image_Size.py`
        - `MLP_and_Linear_Classifiers.py`
        - `CNN_and_MLP_and_Linear_CLassifier.ipynb`
        - `README.md`  <!-- Documentation for neural networks -->
    - **Datasets/**
      - **Original Datasets/**
      - **preprocessed_data/**
      - **archive/**

## Data Preprocessing

Data preprocessing involved cleaning and preparing the datasets to ensure they are suitable for analysis and modeling. Key steps included handling missing values, standardizing data types, and feature engineering. For detailed steps and explanations, refer to the **Data Preprocessing README**.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) uncovered patterns, detected anomalies, and tested hypotheses through summary statistics and graphical representations. For detailed analysis and EDA results, refer to the **EDA README**.

## Clustering

Clustering methods (K-Means, DBSCAN, and Birch) were applied to identify natural groupings in the data. Each clustering method’s results and visualizations, including cluster profiles and evaluation metrics, are detailed in the **Clustering README**.

## Basic Classifiers and Decision Trees

This section explores basic classifiers (Decision Trees, k-Nearest Neighbors, and Naive Bayes) to predict customer satisfaction. Decision Trees performed well, providing interpretable insights, while k-NN achieved the best accuracy on delivery datasets. For detailed results, refer to the **Basic Classifiers and Decision Trees README**.

## Neural Networks and CNNs

Advanced machine learning approaches were explored, including Logistic Regression, Multi-Layer Perceptron (MLP), and Convolutional Neural Networks (CNNs). Highlights include:

- **Tabular Data:** MLP outperformed Logistic Regression with high accuracy and consistent results.
- **Image Data (Food101):** ResNet18 achieved strong performance on food image classification, while an Enhanced Classical CNN provided competitive results with a lightweight architecture.

For detailed methodologies, evaluations, and visualizations, refer to the **Neural Networks README**.

## Repository Contents

- **Scripts:** All code scripts for data preprocessing, EDA, clustering, and modeling.
- **Data:** Raw and preprocessed datasets.
- **Results:** Visualizations, cluster profiles, and evaluation metrics.
- **Documentation:** Detailed README files for each section.

## Setup and Installation

To set up and run this project, follow the steps below:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/DMML-Coursework-Food-Delivery-Satisfaction.git
   
2.	**Navigate to the Project Directory
   ```bash
   cd DMML-Coursework-Food-Delivery-Satisfaction
```

3.	**Install Required Dependencies
```bash
   pip install -r requirements.txt
```

4.	Download the Datasets
	•	Download datasets from the links provided in the Datasets and Sources section.
	•	Place the datasets into:
```bash
   CW_Progress/Datasets/Original Datasets/
```

5.	**Run Scripts
	•	Preprocess data:
```bash
   cd CW_Progress/Coursework_Code_Files/Data_Preprocessing
   python Data_Preprocessing.py
```
   
   •	Perform EDA:
```bash
   cd CW_Progress/Coursework_Code_Files/EDA
   python Exploratory_Data_Analysis.py
```

•	Execute clustering, classifiers, or neural networks scripts as needed.

## Key Insights

   1.	EDA:
	   •	Delivery time and ratings significantly influence customer satisfaction.
	   •	Traffic density and weather conditions impact delivery times.
	2.	Clustering:
	   •	K-Means consistently outperformed other clustering methods, identifying distinct customer satisfaction groups.
	3.	Classifiers:
	   •	Decision Trees provided interpretable rules for customer satisfaction prediction.
	   •	k-NN achieved the highest accuracy on delivery datasets.
	4.	Neural Networks:
	   •	MLP outperformed Logistic Regression with ~98% accuracy.
	   •	ResNet18 was effective for food image classification with a validation accuracy of ~78%.

## Future Work

   1.	Tabular Data:
	   •	Explore advanced architectures like TabNet for feature importance insights.
	   •	Investigate feature selection techniques for improved performance.
	2.	Image Data:
	   •	Experiment with EfficientNet or Vision Transformers for better image classification.
	   •	Implement additional data augmentation techniques.
	3.	Clustering:
	   •	Optimize parameters for DBSCAN and Birch to improve clustering performance.
	   •	Incorporate dimensionality reduction techniques like t-SNE or UMAP for visualization.
	4.	Model Optimization:
	   •	Experiment with model pruning and quantization for efficient deployment.
	   •	Utilize automated hyperparameter tuning techniques (e.g., Grid Search, Bayesian Optimization).

## Additional Notes

   •	Ensure compliance with data privacy and usage policies when handling datasets.
	•	Customize scripts to fit specific analysis requirements or to accommodate unique dataset features.

## Contact Information

For questions, feedback, or collaboration, please contact:
	•	Abdallah Alshaqra - [Email](mailto:ama2018@hw.ac.uk)
	•	Kanishka Agarwal - [Email](mailto:ka2021@hw.ac.uk)
	•	Suhaas - [Email](mailto:saisuhaasv04@gmail.com)
	•	Syeda Zainab - [Email](mailto:sz2021@hw.ac.uk)

## Acknowledgments

We extend our thanks to the contributors of the datasets and the open-source community for the tools and resources used in this project.
