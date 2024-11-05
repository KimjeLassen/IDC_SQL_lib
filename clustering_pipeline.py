import logging
import traceback
from db_base import engine
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib

from k_means_cluster import analyze_clusters, find_optimal_clusters
from dbscan_cluster import analyze_dbscan_clusters, plot_k_distance
from hierarchical_cluster import analyze_hierarchical_clusters, plot_dendrogram

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

def safe_start_run(run_name="Clustering_Run"):
    """
    Safely start an MLflow run by ending any active runs first.
    """
    mlflow.end_run()  # Ends any active run
    return mlflow.start_run(run_name=run_name)

def perform_clustering(data, n_clusters, dbscan_eps, dbscan_min_samples):
    """
    Run KMeans, Agglomerative, and DBSCAN clustering on the data and log the models with MLflow.
    
    Parameters
    ----------
    data : DataFrame or array-like
        The data to cluster.
    n_clusters : int
        The number of clusters to form for KMeans and AgglomerativeClustering.
    dbscan_eps : float
        The maximum distance between two samples for DBSCAN to consider them as in the same neighborhood.
    dbscan_min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point in DBSCAN.
    
    Returns
    -------
    tuple
        kmeans_labels : array
            Cluster labels from KMeans clustering.
        hierarchical_labels : array
            Cluster labels from AgglomerativeClustering.
        dbscan_labels : array
            Cluster labels from DBSCAN clustering.
    """
    try:
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        mlflow.log_params(kmeans.get_params())
        mlflow.log_metric("kmeans_inertia", kmeans.inertia_)
        
        # Log KMeans model
        input_example = data.sample(1)
        signature = infer_signature(data, kmeans.predict(data))
        mlflow.sklearn.log_model(kmeans, "kmeans_model", signature=signature, input_example=input_example)
        
        # Agglomerative Clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(data)
        mlflow.log_params(hierarchical.get_params())
        
        # Log Agglomerative Clustering model
        hierarchical_model_path = "hierarchical_model.pkl"
        joblib.dump(hierarchical, hierarchical_model_path)
        mlflow.log_artifact(hierarchical_model_path, artifact_path="models")
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan_labels = dbscan.fit_predict(data)
        mlflow.log_params({"dbscan_eps": dbscan_eps, "dbscan_min_samples": dbscan_min_samples})
        
        # Log DBSCAN model
        dbscan_model_path = "dbscan_model.pkl"
        joblib.dump(dbscan, dbscan_model_path)
        mlflow.log_artifact(dbscan_model_path, artifact_path="models")
        
        return kmeans_labels, hierarchical_labels, dbscan_labels
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("An error occurred during clustering:", exc_info=True)
        mlflow.log_text(error_trace, "clustering_error_trace.txt")
        mlflow.log_param("clustering_error", str(e))
        raise

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

