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
    Analyze and validate the contents of each K-Means cluster by summarizing user privileges.
    """
    cluster_dir = "k_means_clusters"
    os.makedirs(cluster_dir, exist_ok=True)

    # Remove any existing CSV files in the 'k_means_clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, "*.csv")):
        os.remove(file)

    # Define the column used for K-Means cluster labels
    cluster_label_columns = ["k_means_clusters"]

    # Group the data by K-Means clusters
    clusters = binary_access_matrix.groupby("k_means_clusters")

    for cluster_label, cluster_data in clusters:
        logger.info(f"\nK-Means Cluster {cluster_label}:")

        # Drop the cluster label column to focus on role columns
        cluster_privileges = cluster_data.drop(cluster_label_columns, axis=1)

        # Compute the sum and percentage of each role in the cluster
        privilege_sums = cluster_privileges.sum()
        privilege_percentages = (privilege_sums / len(cluster_data)) * 100

        # Identify privileges common to over 50% of users in the cluster
        common_privileges = privilege_percentages[
            privilege_percentages > 50
        ].sort_values(ascending=False)

        logger.info(f"\nNumber of users in cluster: {len(cluster_data)}")
        logger.info("\nCommon privileges (present in over 50% of users):")
        logger.info(common_privileges.to_string())

        # List the top N most common privileges in the cluster
        top_n = 5
        top_roles = privilege_percentages.sort_values(ascending=False).head(top_n)
        logger.info(f"\nTop {top_n} privileges in the cluster:")
        logger.info(top_roles.to_string())

        # Identify roles unique to this cluster (present in 100% of users)
        unique_roles = privilege_percentages[privilege_percentages == 100]
        if not unique_roles.empty:
            logger.info(
                "\nPrivileges unique to this cluster (present in all users of the cluster):"
            )
            logger.info(unique_roles.to_string())

        # Save and log the cluster data as a CSV file
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
