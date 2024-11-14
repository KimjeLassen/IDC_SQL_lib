# app/clustering/kmeans/k_means_cluster.py

import os
import glob
import mlflow
from sklearn.metrics import silhouette_score
import logging
import traceback
from sklearn.cluster import KMeans

# Set up logging
logger = logging.getLogger(__name__)


def analyze_clusters(binary_access_matrix):
    """
    Analyze K-Means clusters without verbose logging.
    """
    cluster_dir = "k_means_clusters"
    os.makedirs(cluster_dir, exist_ok=True)

    # Remove existing CSV files in the 'k_means_clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, "*.csv")):
        os.remove(file)

    # Group the data by K-Means clusters
    clusters = binary_access_matrix.groupby("k_means_clusters")

    for cluster_label, cluster_data in clusters:
        logger.info(f"Processing K-Means Cluster {cluster_label}")

        # Save each cluster's data as a CSV file
        cluster_file = os.path.join(
            cluster_dir, f"k_means_clusters_cluster_{cluster_label}_data.csv"
        )
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="k_means_clusters_data")


def find_optimal_clusters(data, min_clusters, max_clusters):
    """
    Determine the optimal number of clusters for K-Means using the silhouette score.
    """
    highest_silhouette_score = -1
    optimal_n_clusters = min_clusters

    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            # Initialize and fit the K-Means model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)

            # Calculate the silhouette score for the current number of clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
            mlflow.log_metric("silhouette_score", silhouette_avg, step=n_clusters)

            # Update the optimal number of clusters if the score improves
            if silhouette_avg > highest_silhouette_score:
                highest_silhouette_score = silhouette_avg
                optimal_n_clusters = n_clusters
        except Exception as e:
            # Log any errors encountered during the process
            error_trace = traceback.format_exc()
            logger.error(
                f"An error occurred while finding optimal clusters for n_clusters={n_clusters}:",
                exc_info=True,
            )
            mlflow.log_text(
                error_trace,
                f"find_optimal_clusters_error_n_clusters_{n_clusters}.txt",
            )
            mlflow.log_param(f"error_n_clusters_{n_clusters}", str(e))

    # Log the optimal number of clusters and the highest silhouette score
    logger.info(
        f"\nThe optimal number of clusters is: {optimal_n_clusters} with a K-Means silhouette score of {highest_silhouette_score}"
    )
    mlflow.log_param("optimal_n_clusters", optimal_n_clusters)
    mlflow.log_metric("highest_silhouette_score", highest_silhouette_score)

    return optimal_n_clusters
