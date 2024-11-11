import logging
import mlflow
import mlflow.sklearn
from k_means_cluster import analyze_clusters, find_optimal_clusters
from dbscan_cluster import analyze_dbscan_clusters, plot_k_distance
from hierarchical_cluster import analyze_hierarchical_clusters, plot_dendrogram
from data_manipulation import transform_to_binary_matrix, perform_clustering

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_start_run(run_name="Clustering_Run"):
    """
    Safely start an MLflow run by ending any active runs first.
    """
    mlflow.end_run()  # Ends any active run
    return mlflow.start_run(run_name=run_name)

def run_pipeline(df, min_clusters=3, max_clusters=8, sample_fraction=0.1, max_sample_size=500, dbscan_eps=0.5, dbscan_min_samples=5):
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
    dbscan_eps : float, optional
        The maximum distance between two samples for DBSCAN to consider them as in the same neighborhood.
    dbscan_min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.
    
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
                kmeans_labels, hierarchical_labels, dbscan_labels = perform_clustering(
                    binary_access_matrix, optimal_cluster_count, dbscan_eps, dbscan_min_samples)
                
                # Analyze K-Means clusters
                binary_access_matrix['k_means_clusters'] = kmeans_labels
                analyze_clusters(binary_access_matrix)
                
                # Analyze hierarchical clusters
                binary_access_matrix.drop('k_means_clusters', axis=1, inplace=True)
                analyze_hierarchical_clusters(binary_access_matrix, hierarchical_labels)
                
                # Analyze DBSCAN clusters
                binary_access_matrix.drop('hierarchical_cluster', axis=1, inplace=True)
                analyze_dbscan_clusters(binary_access_matrix, dbscan_labels)
                
                # Sampling for dendrogram visualization using hierarchical clusters
                binary_access_matrix['hierarchical_cluster'] = hierarchical_labels
                subset_data = (
                    binary_access_matrix.groupby('hierarchical_cluster')
                    .apply(lambda x: x.sample(frac=sample_fraction, random_state=42) 
                           if len(x) * sample_fraction <= max_sample_size else x.sample(n=max_sample_size, random_state=42))
                    .reset_index(drop=True)
                    .drop('hierarchical_cluster', axis=1)
                )

                # K-distance plot to help determine eps for DBSCAN
                plot_k_distance(binary_access_matrix, dbscan_min_samples)
                plot_dendrogram(subset_data, subset_data.index.tolist())
                
    except Exception as e:
        logger.error("An error occurred in run_pipeline:", exc_info=True)
        mlflow.log_param("run_pipeline_error", str(e))
        raise

