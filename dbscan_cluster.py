from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import glob  
import mlflow
import matplotlib.pyplot as plt
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dbscan_clusters(binary_access_matrix, dbscan_labels):
    """
    Analyze the contents of each DBSCAN cluster to validate their contents.

    Parameters
    ----------
    binary_access_matrix : DataFrame
        The binary access matrix with roles as columns and users as rows.
    dbscan_labels : array-like
        The cluster labels assigned by DBSCAN, where -1 indicates noise points.

    Returns
    -------
    None
    """
    cluster_dir = 'dbscan_clusters'
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Assign the DBSCAN cluster labels to the binary access matrix
    binary_access_matrix['dbscan_cluster'] = dbscan_labels
    
    # Remove any existing CSV files in 'dbscan_clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, '*.csv')):
        os.remove(file)
    
    # Group the data by DBSCAN clusters
    clusters = binary_access_matrix.groupby('dbscan_cluster')
    
    for cluster_label, cluster_data in clusters:
        if cluster_label == -1:
            print("\nDBSCAN Noise Points (label -1):")
        else:
            print(f"\nDBSCAN Cluster {cluster_label}:")

        # Drop the 'dbscan_cluster' column to get only role columns
        cluster_privileges = cluster_data.drop('dbscan_cluster', axis=1)
        
        # Compute the sum of each role in the cluster
        privilege_sums = cluster_privileges.sum()
        # Calculate the percentage of users in the cluster that have each privilege
        privilege_percentages = (privilege_sums / len(cluster_data)) * 100
        
        # Get privileges that are common in the cluster (e.g., present in over 50% of users)
        common_privileges = privilege_percentages[privilege_percentages > 50].sort_values(ascending=False)
        
        print(f"\nNumber of users in cluster: {len(cluster_data)}")
        print("\nCommon privileges (present in over 50% of users):")
        print(common_privileges)
        
        # List the top N privileges
        top_n = 5
        top_roles = privilege_percentages.sort_values(ascending=False).head(top_n)
        print(f"\nTop {top_n} privileges in the cluster:")
        print(top_roles)
        
        # Identify roles unique to this cluster (if any)
        unique_roles = privilege_percentages[privilege_percentages == 100]
        if not unique_roles.empty:
            print("\nPrivileges unique to this cluster (present in all users of the cluster):")
            print(unique_roles)
        
        # Save and log cluster data
        cluster_file = os.path.join(cluster_dir, f"dbscan_cluster_{cluster_label}_data.csv")
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="dbscan_cluster_data")

def plot_k_distance(data, min_samples):
    """
    Plot the k-distance graph to help determine the optimal value for `eps` in DBSCAN.
    
    Parameters
    ----------
    data : DataFrame or array-like
        The dataset to analyze.
    min_samples : int
        The number of nearest neighbors to consider, typically set to the `min_samples` parameter in DBSCAN.
    
    Returns
    -------
    None
    """
    try:
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        # Sort the distances to the k-th nearest neighbor (min_samples-1 because indexing starts at 0)
        k_distances = np.sort(distances[:, min_samples - 1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
        plt.title("K-distance Plot for DBSCAN")
        plt.grid()
        plt.show()
        
        # Log the plot as an artifact in MLflow
        plt.savefig("k_distance_plot.png")
        mlflow.log_artifact("k_distance_plot.png", artifact_path="plots")
        plt.close()  # Close the plot to free memory
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("An error occurred while plotting K-distance:", exc_info=True)
        mlflow.log_text(error_trace, "plot_k_distance_error_trace.txt")
        mlflow.log_param("plot_k_distance_error", str(e))
        raise