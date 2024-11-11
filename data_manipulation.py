import pandas as pd
import mlflow
import joblib
import traceback
from mlflow.models.signature import infer_signature
from connect import logger
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

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