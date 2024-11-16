# clustering_utils.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA

def load_and_preprocess_data(preprocessed_data_dir):
    """
    Load and preprocess preprocessed data for clustering.
    """
    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(preprocessed_data_dir, 'X_train.csv'))
    X_valid = pd.read_csv(os.path.join(preprocessed_data_dir, 'X_valid.csv'))
    X_test = pd.read_csv(os.path.join(preprocessed_data_dir, 'X_test.csv'))

    # Combine datasets
    X_combined = pd.concat([X_train, X_valid, X_test], axis=0).reset_index(drop=True)

    # Identify categorical and numerical features
    categorical_cols = X_combined.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_combined.select_dtypes(include=[np.number]).columns.tolist()

    # Handle categorical features
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Updated parameter
        X_cat = encoder.fit_transform(X_combined[categorical_cols])
        cat_feature_names = encoder.get_feature_names_out(categorical_cols)
        X_cat = pd.DataFrame(X_cat, columns=cat_feature_names)
    else:
        X_cat = pd.DataFrame()

    # Scale numerical features
    if numerical_cols:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_combined[numerical_cols])
        X_num = pd.DataFrame(X_num, columns=numerical_cols)
    else:
        X_num = pd.DataFrame()

    # Combine numerical and categorical features
    if not X_cat.empty and not X_num.empty:
        X_processed = pd.concat([X_num, X_cat], axis=1)
    elif not X_num.empty:
        X_processed = X_num
    else:
        X_processed = X_cat

    return X_processed

def evaluate_clustering(labels, data, method_name, results_dir=None):
    """
    Evaluate clustering results and save evaluation metrics.
    """
    silhouette_avg = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)

    results = (
        f"{method_name} Clustering Evaluation:\n"
        f" - Silhouette Score: {silhouette_avg:.3f}\n"
        f" - Davies-Bouldin Index: {davies_bouldin:.3f}\n"
        f" - Calinski-Harabasz Index: {calinski_harabasz:.3f}\n"
    )
    print(results)

    if results_dir:
        eval_file = os.path.join(results_dir, f"{method_name.lower()}_evaluation.txt")
        with open(eval_file, 'w') as f:
            f.write(results)
        print(f"{method_name} evaluation metrics saved to {eval_file}")

    # Return metrics for comparison
    return {
        'method': method_name,
        'silhouette_score': silhouette_avg,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_index': calinski_harabasz,
    }

def ensure_directory(directory):
    """
    Ensure that the directory exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_silhouette_scores(cluster_range, silhouette_scores, method, results_dir=None):
    """
    Plot silhouette scores for different numbers of clusters.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title(f'{method} Clustering Silhouette Scores')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid()
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"{method.lower()}_silhouette_scores.png"))
    plt.show()

def plot_clusters(data, labels, method_name, results_dir=None):
    """
    Plot clusters using PCA for dimensionality reduction to 2D.
    """
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    cluster_df = pd.DataFrame(data_2d, columns=['PC1', 'PC2'])
    cluster_df['Cluster'] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', palette='tab10', data=cluster_df, legend='full', s=30
    )
    plt.title(f'{method_name} Clustering Visualization (PCA 2D)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"{method_name.lower()}_clusters.png"), bbox_inches='tight')
    plt.show()

def cluster_profile(data, labels, method_name, results_dir=None):
    """
    Generate cluster profiles by calculating mean feature values for each cluster.
    """
    data_with_labels = data.copy()
    data_with_labels['Cluster'] = labels
    cluster_profiles = data_with_labels.groupby('Cluster').mean()

    if results_dir:
        profile_file = os.path.join(results_dir, f"{method_name.lower()}_cluster_profiles.csv")
        cluster_profiles.to_csv(profile_file)
        print(f"{method_name} cluster profiles saved to {profile_file}")

    print(f"\n{method_name} Cluster Profiles:")
    print(cluster_profiles)

def plot_cluster_counts(labels, method_name, results_dir=None):
    """
    Plot the count of data points in each cluster.
    """
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    plt.title(f'{method_name} Cluster Counts')
    plt.xlabel("Cluster")
    plt.ylabel("Number of Data Points")
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"{method_name.lower()}_cluster_counts.png"))
    plt.show()