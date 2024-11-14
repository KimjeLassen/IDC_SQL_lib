# app/clustering/hierarchical/hierarchical_cluster.py

import os
import glob
import mlflow
import logging
import traceback
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Set up logging
logger = logging.getLogger(__name__)


def analyze_hierarchical_clusters(binary_access_matrix, hierarchical_labels):
    """
    Analyze and validate the contents of each hierarchical cluster by summarizing common privileges.
    """
    cluster_dir = "hierarchical_clusters"
    os.makedirs(cluster_dir, exist_ok=True)

    # Assign hierarchical labels to the binary access matrix
    binary_access_matrix["hierarchical_cluster"] = hierarchical_labels

    # Clear any existing CSV files in 'hierarchical_clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, "*.csv")):
        os.remove(file)

    # Define cluster label columns to exclude from analysis
    cluster_label_columns = [
        "k_means_clusters",
        "hierarchical_cluster",
        "dbscan_cluster",
    ]

    # Group data by hierarchical clusters
    clusters = binary_access_matrix.groupby("hierarchical_cluster")

    for cluster_label, cluster_data in clusters:
        logger.info(f"\nHierarchical Cluster {cluster_label}:")

        # Drop all cluster label columns
        cluster_privileges = cluster_data.drop(
            columns=cluster_label_columns, errors="ignore"
        )

        # Compute the sum and percentage of each privilege in the cluster
        privilege_sums = cluster_privileges.sum()
        privilege_percentages = (privilege_sums / len(cluster_data)) * 100

        # Identify privileges common to over 50% of users in the cluster
        common_privileges = privilege_percentages[
            privilege_percentages > 50
        ].sort_values(ascending=False)

        logger.info(f"\nNumber of users in cluster: {len(cluster_data)}")
        logger.info("\nCommon privileges (present in over 50% of users):")
        logger.info(common_privileges.to_string())

        # List the top N privileges in the cluster
        top_n = 5
        top_roles = privilege_percentages.sort_values(ascending=False).head(top_n)
        logger.info(f"\nTop {top_n} privileges in the cluster:")
        logger.info(top_roles.to_string())

        # Identify roles unique to this cluster
        unique_roles = privilege_percentages[privilege_percentages == 100]
        if not unique_roles.empty:
            logger.info(
                "\nPrivileges unique to this cluster (present in all users of the cluster):"
            )
            logger.info(unique_roles.to_string())

        # Save and log each cluster's data
        cluster_file = os.path.join(
            cluster_dir, f"hierarchical_cluster_{cluster_label}_data.csv"
        )
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="hierarchical_cluster_data")


def plot_dendrogram(data, labels):
    """
    Generate a dendrogram to visualize hierarchical clustering structure.
    """
    try:
        # Perform hierarchical clustering and generate linkage matrix
        linked = linkage(data, method="average")

        plt.figure(figsize=(15, 10))  # Increase figure size for readability

        # Plot the dendrogram with options for a clear view
        dendrogram(
            linked,
            orientation="top",
            labels=labels,
            distance_sort="descending",
            show_leaf_counts=False,
            no_labels=True,  # Remove x-axis labels for clarity
            truncate_mode="level",  # Truncate to show top levels
            p=20,  # Show only the top 20 levels for clarity
        )

        plt.title("Dendrogram for Hierarchical Clustering")
        plt.xlabel("Users")
        plt.ylabel("Distance")
        plt.savefig("dendrogram.png")
        mlflow.log_artifact("dendrogram.png", artifact_path="plots")
        plt.show()
        plt.close()
    except Exception as e:
        # Log error and traceback if plotting fails
        error_trace = traceback.format_exc()
        logger.error("An error occurred while plotting dendrogram:", exc_info=True)
        mlflow.log_text(error_trace, "plot_dendrogram_error_trace.txt")
        mlflow.log_param("plot_dendrogram_error", str(e))
        raise
