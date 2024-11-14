# app/clustering/dbscan/dbscan_cluster.py

from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import glob
import mlflow
import logging
import traceback
from kneed import KneeLocator

# Set up logging
logger = logging.getLogger(__name__)


def analyze_dbscan_clusters(binary_access_matrix, dbscan_labels):
    """
    Analyze DBSCAN clusters without verbose logging.
    """
    cluster_dir = "dbscan_clusters"
    os.makedirs(cluster_dir, exist_ok=True)

    # Assign DBSCAN labels to the binary access matrix
    binary_access_matrix["dbscan_cluster"] = dbscan_labels

    # Remove existing CSV files in 'dbscan_clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, "*.csv")):
        os.remove(file)

    # Group data by DBSCAN clusters
    clusters = binary_access_matrix.groupby("dbscan_cluster")

    for cluster_label, cluster_data in clusters:
        logger.info(f"Processing DBSCAN Cluster {cluster_label}")

        # Save each cluster's data as a CSV file
        cluster_file = os.path.join(
            cluster_dir, f"dbscan_cluster_{cluster_label}_data.csv"
        )
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="dbscan_cluster_data")


def calculate_k_distance(data, min_samples):
    """
    Calculate k-distances without generating a plot.
    """
    try:
        data_dense = data.toarray() if hasattr(data, "toarray") else data

        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(data_dense)
        distances, indices = neighbors_fit.kneighbors(data_dense)
        k_distances = np.sort(distances[:, min_samples - 1])

        return k_distances
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("An error occurred while calculating k-distances:", exc_info=True)
        mlflow.log_text(error_trace, "calculate_k_distance_error_trace.txt")
        mlflow.log_param("calculate_k_distance_error", str(e))
        raise


def detect_eps(k_distances):
    """
    Estimate the optimal `eps` value for DBSCAN by detecting the "elbow" point in the k-distance plot.
    """
    indices = np.arange(len(k_distances))
    kneedle = KneeLocator(
        indices, k_distances, S=1.0, curve="convex", direction="increasing"
    )
    elbow_index = kneedle.knee
    if elbow_index is not None:
        eps_estimated = k_distances[elbow_index]
        return eps_estimated
    else:
        return None
