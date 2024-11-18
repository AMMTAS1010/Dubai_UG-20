# run_clustering.py

from clustering_algorithms import (
    kmeans_clustering,
    dbscan_clustering,
    birch_clustering,
)
from clustering_utils import (
    load_and_preprocess_data,
    evaluate_clustering,
    ensure_directory,
)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    # User selection for dataset type
    print("Select Dataset Type:")
    print("1. Delivery Dataset")
    print("2. Restaurant Dataset")
    dataset_choice = input("Enter the dataset number: ")

    if dataset_choice == "1":
        dataset_type = "delivery"
        dataset_name = "Delivery"
    elif dataset_choice == "2":
        dataset_type = "restaurant"
        dataset_name = "Restaurant"
    else:
        print("Invalid dataset choice.")
        return

    # Define the base directory where preprocessed data is stored
    base_dir = '../Datasets/preprocessed_data'  # Adjust the path as needed
    preprocessed_data_dir = os.path.join(base_dir, dataset_type)

    # Load and preprocess data
    X_processed = load_and_preprocess_data(preprocessed_data_dir)

    # Initialize a list to store evaluation metrics for comparison
    comparison_metrics = []

    # User selection for clustering method
    print("Select Clustering Method:")
    print("1. K-Means")
    print("2. DBSCAN")
    print("3. Birch")
    print("4. Run All and Compare")
    clustering_choice = input("Enter the clustering method number: ")

    if clustering_choice == "1":
        print("Running K-Means Clustering...")
        # Create results directory
        results_dir = f"Results/K_Means_Clustering_{dataset_name}"
        ensure_directory(results_dir)
        labels = kmeans_clustering(X_processed, max_clusters=10, results_dir=results_dir)
        metrics = evaluate_clustering(labels, X_processed, "K-Means", results_dir)
        comparison_metrics.append(metrics)
    elif clustering_choice == "2":
        print("Running DBSCAN Clustering with parameter tuning...")
        results_dir = f"Results/DBScan_Clustering_{dataset_name}"
        ensure_directory(results_dir)
        if dataset_type == "delivery": 
            min_samples_values=[5, 10, 15] 
        else: 
            min_samples_values=[5, 10, 15],
        labels = dbscan_clustering(
            X_processed,
            eps_values=[0.3, 0.5, 0.7, 0.9],
            results_dir=results_dir
        )
        if labels is not None:
            non_noise_mask = labels != -1
            if len(set(labels[non_noise_mask])) > 1:
                metrics = evaluate_clustering(labels[non_noise_mask], X_processed.iloc[non_noise_mask], "DBSCAN", results_dir)
                comparison_metrics.append(metrics)
            else:
                print("DBSCAN did not find sufficient clusters to evaluate.")
        else:
            print("DBSCAN did not find suitable parameters.")
    elif clustering_choice == "3":
        print("Running Birch Clustering with parameter tuning...")
        results_dir = f"Results/Birch_Clustering_{dataset_name}"
        ensure_directory(results_dir)
        labels = birch_clustering(
            X_processed,
            threshold_values=[0.3, 0.5, 0.7],
            n_clusters_values=[2, 3, 4, 5, 6],
            results_dir=results_dir
        )
        if labels is not None:
            metrics = evaluate_clustering(labels, X_processed, "Birch", results_dir)
            comparison_metrics.append(metrics)
        else:
            print("Birch did not find suitable parameters.")
    elif clustering_choice == "4":
        print("Running all clustering algorithms for comparison...")
        results_dir = f"Results/Comparison_of_Clustering_{dataset_name}"
        ensure_directory(results_dir)

        # K-Means
        labels_kmeans = kmeans_clustering(X_processed, max_clusters=10, results_dir=results_dir)
        metrics_kmeans = evaluate_clustering(labels_kmeans, X_processed, "K-Means", results_dir)
        comparison_metrics.append(metrics_kmeans)

        # DBSCAN
        labels_dbscan = dbscan_clustering(
            X_processed,
            eps_values=[0.3, 0.5, 0.7, 0.9],
            min_samples_values=[5, 10, 15],
            results_dir=results_dir
        )
        if labels_dbscan is not None:
            non_noise_mask = labels_dbscan != -1
            if len(set(labels_dbscan[non_noise_mask])) > 1:
                metrics_dbscan = evaluate_clustering(labels_dbscan[non_noise_mask], X_processed.iloc[non_noise_mask], "DBSCAN", results_dir)
                comparison_metrics.append(metrics_dbscan)
            else:
                print("DBSCAN did not find sufficient clusters to evaluate.")
        else:
            print("DBSCAN did not find suitable parameters.")

        # Birch
        labels_birch = birch_clustering(
            X_processed,
            threshold_values=[0.3, 0.5, 0.7],
            n_clusters_values=[2, 3, 4, 5, 6],
            results_dir=results_dir
        )
        if labels_birch is not None:
            metrics_birch = evaluate_clustering(labels_birch, X_processed, "Birch", results_dir)
            comparison_metrics.append(metrics_birch)
        else:
            print("Birch did not find suitable parameters.")

        # Compare algorithms
        compare_algorithms(comparison_metrics, results_dir)
    else:
        print("Invalid clustering method choice.")
        return

    plt.show()  # Show all plots

def compare_algorithms(comparison_metrics, results_dir):
    """
    Compare clustering algorithms based on evaluation metrics.
    """
    df_metrics = pd.DataFrame(comparison_metrics)
    print("\nComparison of Clustering Algorithms:")
    print(df_metrics[['method', 'silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_index']])

    # Save comparison table
    if results_dir:
        df_metrics.to_csv(os.path.join(results_dir, "clustering_algorithms_comparison.csv"), index=False)
        print(f"Comparison metrics saved to {os.path.join(results_dir, 'clustering_algorithms_comparison.csv')}")

    # Plot comparison of Silhouette Scores
    plt.figure(figsize=(8, 5))
    sns.barplot(x='method', y='silhouette_score', data=df_metrics)
    plt.title("Comparison of Clustering Algorithms - Silhouette Score")
    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Silhouette Score")
    if results_dir:
        plt.savefig(os.path.join(results_dir, "clustering_algorithms_silhouette_comparison.png"))
    plt.show()

    # Plot comparison of Davies-Bouldin Index (lower is better)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='method', y='davies_bouldin_index', data=df_metrics)
    plt.title("Comparison of Clustering Algorithms - Davies-Bouldin Index")
    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Davies-Bouldin Index")
    if results_dir:
        plt.savefig(os.path.join(results_dir, "clustering_algorithms_davies_bouldin_comparison.png"))
    plt.show()

    # Plot comparison of Calinski-Harabasz Index (higher is better)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='method', y='calinski_harabasz_index', data=df_metrics)
    plt.title("Comparison of Clustering Algorithms - Calinski-Harabasz Index")
    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Calinski-Harabasz Index")
    if results_dir:
        plt.savefig(os.path.join(results_dir, "clustering_algorithms_calinski_harabasz_comparison.png"))
    plt.show()

if __name__ == "__main__":
    main()