# clustering_algorithms.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from clustering_utils import (
    plot_silhouette_scores,
    plot_clusters,
    cluster_profile,
    plot_cluster_counts,
    plot_birch_dendrogram,  # Moved dendrogram plotting to clustering_utils.py
)

def kmeans_clustering(data, max_clusters=10, results_dir=None):
    sse = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Plot Inertia (Elbow Method)
    plt.figure(figsize=(10, 6))  # Increased figure size
    plt.plot(cluster_range, sse, marker='o')
    plt.title("K-Means Elbow Method", fontsize=14)
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia (SSE)", fontsize=12)
    plt.xticks(cluster_range, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    if results_dir:
        plt.savefig(os.path.join(results_dir, "kmeans_elbow_method.png"), bbox_inches='tight')
    plt.show()

    # Plot Silhouette Scores
    plot_silhouette_scores(cluster_range, silhouette_scores, "K-Means", results_dir)

    # Choose optimal k
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because range starts at 2
    print(f"Optimal number of clusters determined by silhouette score: {optimal_k}")

    # Fit KMeans with optimal_k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(data)

    # Save cluster labels
    cluster_results = pd.DataFrame(labels, columns=['Cluster'])
    if results_dir:
        cluster_results.to_csv(os.path.join(results_dir, "kmeans_clusters.csv"), index=False)
        print("K-Means clustering results saved.")

    # Plot clusters
    plot_clusters(data, labels, "K-Means", results_dir)
    plot_cluster_counts(labels, "K-Means", results_dir)

    # Generate cluster profiles
    cluster_profile(data, labels, "K-Means", results_dir)

    return labels

def dbscan_clustering(data, eps_values=[0.3, 0.5, 0.7, 0.9], min_samples_values=[5, 10, 15], results_dir=None):
    best_score = -1
    best_eps = None
    best_min_samples = None
    best_labels = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                non_noise_mask = labels != -1
                try:
                    score = silhouette_score(data[non_noise_mask], labels[non_noise_mask])
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                        best_labels = labels
                except:
                    continue

    if best_labels is not None:
        print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")
        print(f"Best Silhouette Score: {best_score:.3f}")
        labels = best_labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Save cluster labels
        cluster_results = pd.DataFrame(labels, columns=['Cluster'])
        if results_dir:
            cluster_results.to_csv(os.path.join(results_dir, "dbscan_clusters.csv"), index=False)
            print("DBSCAN clustering results saved.")

        # Plot clusters (excluding noise)
        non_noise_mask = labels != -1
        if n_clusters > 1:
            plot_clusters(data[non_noise_mask], labels[non_noise_mask], "DBSCAN", results_dir)
            plot_cluster_counts(labels[non_noise_mask], "DBSCAN", results_dir)
            # Generate cluster profiles
            cluster_profile(data.iloc[non_noise_mask], labels[non_noise_mask], "DBSCAN", results_dir)
        else:
            print("Skipping cluster plot for DBSCAN due to insufficient clusters.")
    else:
        print("DBSCAN did not find sufficient clusters with the given parameter ranges.")
        labels = None

    return labels

def birch_clustering(data, threshold_values=[0.3, 0.5, 0.7], n_clusters_values=[2, 3, 4, 5, 6], results_dir=None):
    best_score = -1
    best_threshold = None
    best_n_clusters = None
    best_labels = None
    best_model = None

    for threshold in threshold_values:
        for n_clusters in n_clusters_values:
            birch_model = Birch(threshold=threshold, n_clusters=n_clusters)
            labels = birch_model.fit_predict(data)
            unique_labels = set(labels)
            n_clusters_estimated = len(unique_labels)

            if n_clusters_estimated > 1:
                try:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        best_n_clusters = n_clusters
                        best_labels = labels
                        best_model = birch_model
                except:
                    continue

    if best_labels is not None:
        print(f"Best Birch parameters: threshold={best_threshold}, n_clusters={best_n_clusters}")
        print(f"Best Silhouette Score: {best_score:.3f}")
        labels = best_labels

        # Save cluster labels
        cluster_results = pd.DataFrame(labels, columns=['Cluster'])
        if results_dir:
            cluster_results.to_csv(os.path.join(results_dir, "birch_clusters.csv"), index=False)
            print("Birch clustering results saved.")

        # Plot clusters
        plot_clusters(data, labels, "Birch", results_dir)
        plot_cluster_counts(labels, "Birch", results_dir)

        # Generate cluster profiles
        cluster_profile(data, labels, "Birch", results_dir)

        # Plot dendrogram
        plot_birch_dendrogram(best_model, data, "Birch", results_dir)
    else:
        print("Birch did not find sufficient clusters with the given parameter ranges.")
        labels = None

    return labels