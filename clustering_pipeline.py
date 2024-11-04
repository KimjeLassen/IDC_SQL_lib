import os
import glob  
import logging
import traceback
from db_base import engine
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(sql_query):
    """
    Fetch data from the database using the provided SQL query.
    
    Parameters
    ----------
    sql_query : str
        The SQL query string to execute.
    
    Returns
    -------
    DataFrame or None
        The data retrieved from the database as a pandas DataFrame,
        or None if an error occurs.
    """
    try:
        df = pd.read_sql(sql_query, engine)
        logger.info("Data loaded successfully from the database.")
        # Log data shape
        mlflow.log_param("data_shape", df.shape)
        return df
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("An error occurred while fetching data:", exc_info=True)
        mlflow.log_text(error_trace, "fetch_error_trace.txt")
        mlflow.log_param("fetch_error", str(e))
        return None

def transform_to_binary_matrix(df):
    """
    Transform system roles into a binary access matrix.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame containing 'user_id' and 'system_role_name' columns.
    
    Returns
    -------
    DataFrame
        A binary access matrix where rows represent users and columns represent system roles.
        Each cell contains 1 if the user has the role, and 0 otherwise.
    """
    # Convert 'system_role_name' to dummy variables (one-hot encoding)
    binary_access_matrix = pd.get_dummies(df, columns=['system_role_name'], prefix='', prefix_sep='')
    # Group by 'user_id' and take the maximum to combine multiple roles per user
    binary_access_matrix = binary_access_matrix.groupby('user_id').max()
    # Log the number of unique users and roles
    mlflow.log_param("unique_users", binary_access_matrix.shape[0])
    mlflow.log_param("unique_roles", binary_access_matrix.shape[1])
    return binary_access_matrix

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
    cluster_dir = 'clusters'
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Remove existing CSV files in 'clusters' directory
    for file in glob.glob(os.path.join(cluster_dir, '*.csv')):
        os.remove(file)
    
    # Group the data by clusters
    clusters = binary_access_matrix.groupby('cluster')
    
    for cluster_label, cluster_data in clusters:
        print(f"\nCluster {cluster_label}:")
        # Remove 'cluster' column to get only role columns
        cluster_privileges = cluster_data.drop('cluster', axis=1)
        
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
        cluster_file = os.path.join(cluster_dir, f"cluster_{cluster_label}_data.csv")
        cluster_data.to_csv(cluster_file)
        mlflow.log_artifact(cluster_file, artifact_path="cluster_data")

def find_optimal_clusters(data, min_clusters=2, max_clusters=10):
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

def safe_start_run(run_name="Clustering_Run"):
    """
    Safely start an MLflow run by ending any active runs first.
    """
    mlflow.end_run()  # Ends any active run
    return mlflow.start_run(run_name=run_name)

def perform_clustering(data, n_clusters):
    """
    Run KMeans clustering on the data and log the model with MLflow. Also perform AgglomerativeClustering and save it as an artifact.
    
    Parameters
    ----------
    data : DataFrame or array-like
        The data to cluster.
    n_clusters : int
        The number of clusters to form.
    
    Returns
    -------
    tuple
        kmeans_labels : array
            Cluster labels from KMeans clustering.
        hierarchical_labels : array
            Cluster labels from AgglomerativeClustering.
    """
    try:
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        mlflow.log_params(kmeans.get_params())
        mlflow.log_metric("kmeans_inertia", kmeans.inertia_)
        
        # Define model signature and input example for KMeans
        input_example = data.sample(1)
        signature = infer_signature(data, kmeans.predict(data))
        
        # Log KMeans model with signature and input example
        mlflow.sklearn.log_model(kmeans, "kmeans_model", signature=signature, input_example=input_example)
        
        # Agglomerative Clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(data)
        mlflow.log_params(hierarchical.get_params())
        
        # Save AgglomerativeClustering model as an artifact
        hierarchical_model_path = "hierarchical_model.pkl"
        joblib.dump(hierarchical, hierarchical_model_path)
        mlflow.log_artifact(hierarchical_model_path, artifact_path="models")
        
        return kmeans_labels, hierarchical_labels
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("An error occurred during clustering:", exc_info=True)
        mlflow.log_text(error_trace, "clustering_error_trace.txt")
        mlflow.log_param("clustering_error", str(e))
        raise

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
            p=5  # Show only the top 5 levels
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

def run_pipeline(df, min_clusters=2, max_clusters=10, sample_fraction=0.1, max_sample_size=500):
    """
    Execute the full data processing and clustering pipeline.
    
    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing 'user_id' and 'system_role_name' columns.
    min_clusters : int, optional
        The minimum number of clusters to try. Default is 2.
    max_clusters : int, optional
        The maximum number of clusters to try. Default is 10.
    sample_fraction : float, optional
        Fraction of samples to take for dendrogram visualization. Default is 0.1 (10%).
    max_sample_size : int, optional
        Maximum number of samples per cluster for dendrogram visualization. Default is 500.
    
    Returns
    -------
    None
    """
    mlflow.set_experiment("Role Mining Clustering Experiment")
    try:
        with safe_start_run(run_name="Clustering_Run") as run:
            # Log pipeline parameters
            mlflow.log_param("min_clusters", min_clusters)
            mlflow.log_param("max_clusters", max_clusters)
            mlflow.log_param("sample_fraction", sample_fraction)
            mlflow.log_param("max_sample_size", max_sample_size)
            mlflow.log_param("random_state", 42)
            mlflow.set_tag("pipeline", "Role Mining Clustering")
            
            if df is not None:
                binary_access_matrix = transform_to_binary_matrix(df)
                optimal_cluster_count = find_optimal_clusters(binary_access_matrix, min_clusters, max_clusters)
                kmeans_labels, hierarchical_labels = perform_clustering(binary_access_matrix, optimal_cluster_count)
                binary_access_matrix['cluster'] = kmeans_labels
                analyze_clusters(binary_access_matrix)
                
                subset_data = (
                    binary_access_matrix.groupby('cluster')
                    .apply(lambda x: x.sample(frac=sample_fraction, random_state=42) if len(x) * sample_fraction <= max_sample_size else x.sample(n=max_sample_size, random_state=42))
                    .reset_index(drop=True)
                    .drop('cluster', axis=1)
                )
                
                plot_dendrogram(subset_data, subset_data.index.tolist())
    except Exception as e:
        logger.error("An error occurred in run_pipeline:", exc_info=True)
        mlflow.log_param("run_pipeline_error", str(e))
        raise
