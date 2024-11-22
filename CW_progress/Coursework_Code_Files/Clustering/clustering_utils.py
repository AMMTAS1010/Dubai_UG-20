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
from scipy.cluster.hierarchy import dendrogram, linkage  # Added import for dendrogram

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
        # Updated parameter: Replace 'sparse=False' with 'sparse_output=False'
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
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
    plt.figure(figsize=(10, 6))  # Increased figure size
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title(f'{method} Clustering Silhouette Scores', fontsize=14)
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(cluster_range, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"{method.lower()}_silhouette_scores.png"), bbox_inches='tight')
    plt.show()

def plot_clusters(data, labels, method_name, results_dir=None):
    """
    Plot clusters using PCA for dimensionality reduction to 2D.
    The legend is placed below the plot, spanning maximum width without exceeding image width.
    """
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    cluster_df = pd.DataFrame(data_2d, columns=['PC1', 'PC2'])
    cluster_df['Cluster'] = labels

    unique_clusters = cluster_df['Cluster'].nunique()
    
    # Determine if the method is DBSCAN
    is_dbscan = method_name.lower() == 'dbscan'
    
    # Fixed figure size
    plt.figure(figsize=(10, 8))
    
    # Choose a color palette that can handle more colors if necessary
    if unique_clusters > 10:
        palette = sns.color_palette("hsv", unique_clusters)
    else:
        palette = 'tab10'
    
    scatter = sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', palette=palette, data=cluster_df, legend=False, s=30
    )
    plt.title(f'{method_name} Clustering Visualization (PCA 2D)', fontsize=14)
    
    # Create a separate legend below the plot
    handles, _ = scatter.get_legend_handles_labels()
    # Determine the number of columns for the legend
    if unique_clusters <= 10:
        ncol = 2
    elif unique_clusters <= 20:
        ncol = 4
    elif unique_clusters <= 50:
        ncol = 6
    else:
        ncol = 10  # Max columns to prevent exceeding image width

    plt.legend(handles=handles, labels=cluster_df['Cluster'].unique(),
               title='Cluster', fontsize=10, title_fontsize=12,
               bbox_to_anchor=(0.5, -0.15), loc='upper center',
               ncol=ncol, borderaxespad=0.)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for the legend
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
    Use a logarithmic scale for better readability of large count disparities.
    """
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    num_clusters = cluster_counts.shape[0]
    
    # Determine if we should display value labels
    display_value_labels = num_clusters <= 100  # Adjust this threshold as needed

    # Switch to horizontal bar plot for better readability with many clusters
    plt.figure(figsize=(12, max(8, num_clusters * 0.02)))  # Adjust height based on number of clusters
    sns.set_style("whitegrid")
    
    # Use a single, solid color for all bars to enhance visibility
    bar_plot = sns.barplot(
        x=cluster_counts.values, 
        y=cluster_counts.index.astype(str), 
        color='steelblue',  # Solid color for visibility
    )
    
    plt.title(f'{method_name} Cluster Counts', fontsize=14)
    plt.xlabel("Number of Data Points (Log Scale)", fontsize=12)
    plt.ylabel("Cluster", fontsize=12)
    
    # Apply logarithmic scale for better readability
    plt.xscale("log")
    plt.grid(axis='x', linestyle='--', linewidth=0.5)  # Add gridlines for x-axis

    # Limit the number of y-ticks to prevent label clutter
    if num_clusters > 100:
        # Show labels for every 50th cluster
        tick_positions = range(0, num_clusters, 50)
        tick_labels = [str(cluster_counts.index[i]) if i < num_clusters else "" for i in tick_positions]
        bar_plot.set_yticks(tick_positions)
        bar_plot.set_yticklabels(tick_labels)
    elif num_clusters > 50:
        # Show labels for every 10th cluster
        tick_positions = range(0, num_clusters, 10)
        tick_labels = [str(cluster_counts.index[i]) if i < num_clusters else "" for i in tick_positions]
        bar_plot.set_yticks(tick_positions)
        bar_plot.set_yticklabels(tick_labels)
    else:
        # Show all labels
        bar_plot.set_yticks(range(num_clusters))
        bar_plot.set_yticklabels([str(cluster) for cluster in cluster_counts.index], fontsize=8)

    # Adjust y-axis label font size
    plt.yticks(fontsize=6)

    # Add value labels on the bars if the number of clusters is manageable
    if display_value_labels:
        for index, value in enumerate(cluster_counts.values):
            plt.text(value + max(cluster_counts.values)*0.01, index, str(value), va='center', fontsize=6)

    plt.tight_layout()  # Adjust layout to prevent clipping
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"{method_name.lower()}_cluster_counts.png"), bbox_inches='tight')
    plt.show()

def plot_birch_dendrogram(birch_model, data, method_name, results_dir=None):
    """
    Plot a dendrogram for the Birch clustering.

    Parameters:
    - birch_model: The trained Birch model.
    - data: The dataset used for clustering.
    - method_name: Name of the clustering method (e.g., "Birch").
    - results_dir: Directory to save the dendrogram plot.
    """
    # Extract the subcluster centroids
    subcluster_centers = birch_model.subcluster_centers_

    # Perform hierarchical clustering on the subcluster centroids
    linked = linkage(subcluster_centers, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(15, 8))  # Increased figure size to accommodate more labels
    dendrogram(
        linked,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        no_labels=True,  # Hide leaf labels to prevent overlap
    )
    plt.title(f'{method_name} Clustering Dendrogram', fontsize=16)
    plt.xlabel('Subcluster Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()  # Adjust layout to prevent clipping
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"{method_name.lower()}_dendrogram.png"), bbox_inches='tight')
        print(f"{method_name} dendrogram plot saved.")
    plt.show()