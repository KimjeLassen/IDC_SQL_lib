# dbscan_cluster.py
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import glob
import mlflow
import matplotlib.pyplot as plt
import logging
import traceback
from kneed import KneeLocator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dbscan_clusters(binary_access_matrix, dbscan_labels):
    """
    Analyze and validate each DBSCAN cluster's contents by summarizing the privileges present in each cluster.

    Parameters
    ----------
    binary_access_matrix : DataFrame
        The binary access matrix where each row represents a user and each column represents a role.
    dbscan_labels : array-like
        The cluster labels assigned by DBSCAN, where -1 indicates noise points (outliers).

    Returns
    -------
    None

    MLflow Logging
    --------------
    Saves each cluster's data to CSV files, which are logged as artifacts.

    Details
    -------
    - For each cluster (including noise points), calculates:
        - Total number of users.
        - Percentage of users in the cluster that possess each privilege.
        - Privileges common to more than 50% of the cluster.
        - Top 5 most common privileges.
        - Privileges unique to the cluster (if present in 100% of users).
    """
    cluster_dir = "dbscan_clusters"
    os.makedirs(cluster_dir, exist_ok=True)

    # Assign DBSCAN labels to the binary access matrix
    binary_access_matrix["dbscan_cluster"] = dbscan_labels

    # Remove existing CSV files in 'dbscan_clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, "*.csv")):
        os.remove(file)

    # Define cluster label columns to drop for analysis
    cluster_label_columns = [
        "k_means_clusters",
        "hierarchical_cluster",
        "dbscan_cluster",
    ]

    # Group data by DBSCAN clusters
    clusters = binary_access_matrix.groupby("dbscan_cluster")

    for cluster_label, cluster_data in clusters:
        if cluster_label == -1:
            print("\nDBSCAN Noise Points (label -1):")
        else:
            print(f"\nDBSCAN Cluster {cluster_label}:")

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

        print(f"\nNumber of users in cluster: {len(cluster_data)}")
        print("\nCommon privileges (present in over 50% of users):")
        print(common_privileges)

        # List the top N privileges in the cluster
        top_n = 5
        top_roles = privilege_percentages.sort_values(ascending=False).head(top_n)
        print(f"\nTop {top_n} privileges in the cluster:")
        print(top_roles)

        # Identify roles unique to this cluster
        unique_roles = privilege_percentages[privilege_percentages == 100]
        if not unique_roles.empty:
            print(
                "\nPrivileges unique to this cluster (present in all users of the cluster):"
            )
            print(unique_roles)

        # Save and log each cluster's data
        cluster_file = os.path.join(
            cluster_dir, f"dbscan_cluster_{cluster_label}_data.csv"
        )
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="dbscan_cluster_data")


def plot_k_distance(data, min_samples):
    """
    Generate a k-distance plot to help estimate the optimal `eps` value for DBSCAN.

    Parameters
    ----------
    data : DataFrame or array-like
        The dataset to analyze.
    min_samples : int
        Number of nearest neighbors to consider, typically set to the `min_samples` parameter in DBSCAN.

    Returns
    -------
    k_distances : array
        Array of sorted distances to the `min_samples`-th nearest neighbor for each point.

    MLflow Logging
    --------------
    Saves the k-distance plot as an artifact.

    Details
    -------
    - Uses NearestNeighbors to calculate distances.
    - Plots distances sorted in ascending order to visualize the "elbow," which indicates the optimal `eps`.
    """
    try:
        # Convert sparse matrix to dense if needed
        data_dense = data.toarray() if hasattr(data, "toarray") else data

        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(data_dense)
        distances, indices = neighbors_fit.kneighbors(data_dense)
        k_distances = np.sort(distances[:, min_samples - 1])

        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
        plt.title("K-distance Plot for DBSCAN")
        plt.grid()
        plt.savefig("k_distance_plot.png")
        mlflow.log_artifact("k_distance_plot.png", artifact_path="plots")
        plt.show()
        plt.close()

        return k_distances
    except Exception as e:
        # Log error and traceback if plotting fails
        error_trace = traceback.format_exc()
        logger.error("An error occurred while plotting K-distance:", exc_info=True)
        mlflow.log_text(error_trace, "plot_k_distance_error_trace.txt")
        mlflow.log_param("plot_k_distance_error", str(e))
        raise


def detect_eps(k_distances):
    """
    Estimate the optimal `eps` value for DBSCAN by detecting the "elbow" point in the k-distance plot.

    Parameters
    ----------
    k_distances : array
        Sorted distances to the `min_samples`-th nearest neighbor, typically output from `plot_k_distance`.

    Returns
    -------
    float or None
        Estimated `eps` value if an elbow is detected; otherwise, None.

    Details
    -------
    - Uses the KneeLocator to identify the "knee" (elbow) point, where the curve shows the most significant change in slope.
    - The detected `eps` is generally the value at the knee index in `k_distances`.
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
