# data_manipulation.py
import pandas as pd
import mlflow
import joblib
import traceback
from mlflow.models.signature import infer_signature
from connect import logger
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


def transform_to_binary_matrix(df):
    """
    Convert a DataFrame of user-role mappings into a binary access matrix.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing 'user_id' and 'system_role_name' columns.

    Returns
    -------
    DataFrame
        A binary access matrix where each row represents a user and each column represents a role.
        Cells contain 1 if the user has the role and 0 otherwise.

    MLflow Logging
    --------------
    Logs the number of unique users and roles as parameters.
    """
    # Convert 'system_role_name' to dummy variables (one-hot encoding)
    binary_access_matrix = pd.get_dummies(
        df, columns=["system_role_name"], prefix="", prefix_sep=""
    )
    # Group by 'user_id' and take the maximum to aggregate multiple roles per user
    binary_access_matrix = binary_access_matrix.groupby("user_id").max()
    # Log the number of unique users and roles
    mlflow.log_param("unique_users", binary_access_matrix.shape[0])
    mlflow.log_param("unique_roles", binary_access_matrix.shape[1])
    return binary_access_matrix


def perform_kmeans_and_hierarchical(data, n_clusters):
    """
    Apply KMeans and Agglomerative Clustering to TF-IDF-transformed data and log the results.

    Parameters
    ----------
    data : array-like
        The TF-IDF-transformed data to be clustered.
    n_clusters : int
        The number of clusters to form for both clustering methods.

    Returns
    -------
    tuple
        kmeans_labels : array
            Cluster labels assigned by KMeans.
        hierarchical_labels : array
            Cluster labels assigned by Agglomerative Clustering.

    MLflow Logging
    --------------
    - Logs model parameters and silhouette scores for each clustering method.
    - Saves the KMeans and Agglomerative models as artifacts.

    Raises
    ------
    Exception
        If any error occurs during clustering, it logs the traceback and raises the error.
    """
    try:
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        mlflow.log_params(kmeans.get_params())
        mlflow.log_metric("kmeans_inertia", kmeans.inertia_)

        # Log the KMeans model to MLflow
        input_example = data[:1]
        signature = infer_signature(data, kmeans.predict(data))
        mlflow.sklearn.log_model(
            kmeans, "kmeans_model", signature=signature, input_example=input_example
        )

        # Agglomerative Clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(data)
        mlflow.log_params(hierarchical.get_params())

        # Calculate and log the silhouette score for Agglomerative Clustering
        silhouette_avg = silhouette_score(data, hierarchical_labels)
        mlflow.log_metric("hierarchical_silhouette_score", silhouette_avg)
        print(f"Agglomerative Clustering Silhouette Score: {silhouette_avg}")

        # Save the Agglomerative model as an artifact
        hierarchical_model_path = "hierarchical_model.pkl"
        joblib.dump(hierarchical, hierarchical_model_path)
        mlflow.log_artifact(hierarchical_model_path, artifact_path="models")

        return kmeans_labels, hierarchical_labels
    except Exception as e:
        # Log error and traceback if clustering fails
        error_trace = traceback.format_exc()
        logger.error(
            "An error occurred during K-Means and Agglomerative clustering:",
            exc_info=True,
        )
        mlflow.log_text(error_trace, "kmeans_hierarchical_clustering_error_trace.txt")
        mlflow.log_param("kmeans_hierarchical_clustering_error", str(e))
        raise


def perform_dbscan(data, dbscan_eps, dbscan_min_samples):
    """
    Apply DBSCAN clustering to data, estimate silhouette scores if applicable, and log the results.

    Parameters
    ----------
    data : array-like
        The binary or numerical data to be clustered with DBSCAN.
    dbscan_eps : float
        The maximum distance between two samples to be considered in the same neighborhood.
    dbscan_min_samples : int
        The number of samples in a neighborhood for a point to be considered a core point.

    Returns
    -------
    array
        Cluster labels assigned by DBSCAN, where -1 indicates noise points.

    MLflow Logging
    --------------
    - Logs DBSCAN parameters and silhouette scores (if there are sufficient clusters).
    - Saves the DBSCAN model as an artifact.

    Raises
    ------
    Exception
        If any error occurs during clustering, it logs the traceback and raises the error.
    """
    try:
        # DBSCAN clustering
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan_labels = dbscan.fit_predict(data)
        mlflow.log_params(
            {"dbscan_eps": dbscan_eps, "dbscan_min_samples": dbscan_min_samples}
        )

        # Calculate and log the silhouette score if there are sufficient clusters
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        if n_clusters > 1:
            core_samples_mask = dbscan_labels != -1
            dbscan_silhouette = silhouette_score(
                data[core_samples_mask], dbscan_labels[core_samples_mask]
            )
            mlflow.log_metric("dbscan_silhouette_score", dbscan_silhouette)
            print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
        else:
            print("DBSCAN did not find enough clusters to compute silhouette score.")
            mlflow.log_metric("dbscan_silhouette_score", float("nan"))

        # Save the DBSCAN model as an artifact
        dbscan_model_path = "dbscan_model.pkl"
        joblib.dump(dbscan, dbscan_model_path)
        mlflow.log_artifact(dbscan_model_path, artifact_path="models")

        return dbscan_labels
    except Exception as e:
        # Log error and traceback if clustering fails
        error_trace = traceback.format_exc()
        logger.error("An error occurred during DBSCAN clustering:", exc_info=True)
        mlflow.log_text(error_trace, "dbscan_clustering_error_trace.txt")
        mlflow.log_param("dbscan_clustering_error", str(e))
        raise
