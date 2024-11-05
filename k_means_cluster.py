import os
import glob
import mlflow
from sklearn.metrics import silhouette_score
import logging
import traceback
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_clusters(binary_access_matrix):
    """
    Analyze the contents of each cluster to validate their contents.
    
    Parameters
    ----------
    binary_access_matrix : DataFrame
        The binary access matrix with cluster labels assigned.
    
    Returns
    -------
    None
    """
    cluster_dir = 'k_means_clusters'
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Remove existing CSV files in 'clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, '*.csv')):
        os.remove(file)
    
    # Group the data by clusters
    clusters = binary_access_matrix.groupby('k_means_clusters')
    
    for cluster_label, cluster_data in clusters:
        print(f"\nK-Means Cluster {cluster_label}:")
        # Remove 'cluster' column to get only role columns
        cluster_privileges = cluster_data.drop('k_means_clusters', axis=1)
        
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
        cluster_file = os.path.join(cluster_dir, f"k_means_clusters_cluster_{cluster_label}_data.csv")
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="k_means_clusters_data")

def find_optimal_clusters(data, min_clusters, max_clusters):
    """
    Determine the optimal number of clusters using the silhouette score.
    
    Parameters
    ----------
    data : DataFrame or array-like
        The data to cluster.
    min_clusters : int, optional
        The minimum number of clusters to try. Default is 2.
    max_clusters : int, optional
        The maximum number of clusters to try. Default is 10.
    
    Returns
    -------
    int
        The optimal number of clusters based on the highest silhouette score.
    """
    silhouette_scores = []
    highest_silhouette_score = -1
    optimal_n_clusters = min_clusters
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            # Log silhouette score with step as n_clusters
            mlflow.log_metric("silhouette_score", silhouette_avg, step=n_clusters)
            
            if silhouette_avg > highest_silhouette_score:
                highest_silhouette_score = silhouette_avg
                optimal_n_clusters = n_clusters
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"An error occurred while finding optimal clusters for n_clusters={n_clusters}:", exc_info=True)
            mlflow.log_text(error_trace, f"find_optimal_clusters_error_n_clusters_{n_clusters}.txt")
            mlflow.log_param(f"error_n_clusters_{n_clusters}", str(e))
    
    print(f"\nThe optimal number of clusters is: {optimal_n_clusters} with a silhouette score of {highest_silhouette_score}")
    mlflow.log_param("optimal_n_clusters", optimal_n_clusters)
    mlflow.log_metric("highest_silhouette_score", highest_silhouette_score)
    return optimal_n_clusters