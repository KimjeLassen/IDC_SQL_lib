import os
import glob  
import mlflow
import logging
import traceback
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_hierarchical_clusters(binary_access_matrix, hierarchical_labels):
    """
    Analyze the contents of each hierarchical cluster to validate their contents.

    Parameters
    ----------
    binary_access_matrix : DataFrame
        The binary access matrix with roles as columns and users as rows.
    hierarchical_labels : array-like
        The cluster labels assigned by AgglomerativeClustering.

    Returns
    -------
    None
    """
    cluster_dir = 'hierarchical_clusters'
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Assign the hierarchical cluster labels to the binary access matrix
    binary_access_matrix['hierarchical_cluster'] = hierarchical_labels
    
    # Remove any existing CSV files in 'hierarchical_clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, '*.csv')):
        os.remove(file)
    
    # Group the data by hierarchical clusters
    clusters = binary_access_matrix.groupby('hierarchical_cluster')
    
    for cluster_label, cluster_data in clusters:
        print(f"\nHierarchical Cluster {cluster_label}:")
        
        # Drop the 'hierarchical_cluster' column to get only role columns
        cluster_privileges = cluster_data.drop('hierarchical_cluster', axis=1)
        
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
        cluster_file = os.path.join(cluster_dir, f"hierarchical_cluster_{cluster_label}_data.csv")
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="hierarchical_cluster_data")

def plot_dendrogram(data, labels):
    """
    Generate a dendrogram for hierarchical clustering.
    
    Parameters
    ----------
    data : DataFrame or array-like
        The data to cluster and plot.
    labels : list
        Labels for each data point (e.g., user IDs).
    
    Returns
    -------
    None
    """
    try:
        # Perform hierarchical clustering to obtain linkage matrix
        linked = linkage(data, method='average')
        
        # Adjust the figure size
        plt.figure(figsize=(15, 10))  # Increase figure size
    
        # Plot the dendrogram with options for readability
        dendrogram(
            linked,
            orientation='top',
            labels=labels,
            distance_sort='descending',
            show_leaf_counts=False,
            no_labels=True,  # Remove labels on x-axis
            truncate_mode='level',  # Truncate to top levels
            p=20 # Show only the top 5 levels
        )
        
        plt.title("Dendrogram for Hierarchical Clustering")
        plt.xlabel("Users")
        plt.ylabel("Euclidean Distance")
        # Save the plot to a file
        plt.savefig("dendrogram.png")
        # Log the artifact
        mlflow.log_artifact("dendrogram.png", artifact_path="plots")
        plt.show()
        plt.close()  # Close the plot to free memory
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("An error occurred while plotting dendrogram:", exc_info=True)
        mlflow.log_text(error_trace, "plot_dendrogram_error_trace.txt")
        mlflow.log_param("plot_dendrogram_error", str(e))
        raise