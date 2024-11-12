# app/clustering/clustering_pipeline.py
import logging
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from clustering.kmeans.k_means_cluster import (
    analyze_clusters,
    find_optimal_clusters,
)
from clustering.dbscan.dbscan_cluster import (
    analyze_dbscan_clusters,
    plot_k_distance,
    detect_eps,
)
from clustering.hierarchical.hierarchical_cluster import (
    analyze_hierarchical_clusters,
    plot_dendrogram,
)
from clustering.data_manipulation.data_manipulation import (
    transform_to_binary_matrix,
    perform_dbscan,
    perform_kmeans_and_hierarchical,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_start_run(run_name="Clustering_Run"):
    """
    Safely start an MLflow run by ending any active runs first.

    Parameters
    ----------
    run_name : str, optional
        The name of the MLflow run. Default is "Clustering_Run".

    Returns
    -------
    mlflow.ActiveRun
        The started MLflow run object.
    """
    mlflow.end_run()  # Ends any active run
    return mlflow.start_run(run_name=run_name)


def run_pipeline(
    df,
    min_clusters=3,
    max_clusters=8,
    sample_fraction=0.1,
    max_sample_size=500,
    dbscan_eps=None,
    dbscan_min_samples=5,
):
    """
    Execute the full data processing and clustering pipeline.

    This function performs the following steps:
    - Transforms the input DataFrame into a binary access matrix.
    - Applies TF-IDF transformation for K-Means and Agglomerative Clustering.
    - Uses the binary data for DBSCAN clustering.
    - Finds the optimal number of clusters using the silhouette score.
    - Estimates the `eps` parameter for DBSCAN if not provided.
    - Performs K-Means, Agglomerative, and DBSCAN clustering.
    - Analyzes the clusters and logs the results.
    - Generates a dendrogram for hierarchical clustering.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing 'user_id' and 'system_role_name' columns.
    min_clusters : int, optional
        The minimum number of clusters to try. Default is 3.
    max_clusters : int, optional
        The maximum number of clusters to try. Default is 8.
    sample_fraction : float, optional
        Fraction of samples to take for dendrogram visualization per cluster.
        Default is 0.1 (10%).
    max_sample_size : int, optional
        Maximum number of samples per cluster for dendrogram visualization.
        Default is 500.
    dbscan_eps : float, optional
        The maximum distance between two samples for DBSCAN to consider them as
        in the same neighborhood. If None, `eps` will be estimated automatically
        using the k-distance plot.
    dbscan_min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point in DBSCAN. Default is 5.

    Returns
    -------
    None

    Notes
    -----
    - The function logs various parameters and artifacts using MLflow.
    - It handles exceptions by logging the error details and re-raising the exception.
    - Ensure that the input DataFrame `df` is not empty and contains the required columns.

    """
    mlflow.set_experiment("Role Mining Clustering Experiment")
    try:
        with safe_start_run(run_name="Clustering_Run"):
            # Log pipeline parameters
            # Note: eps is currently being set automatically if not provided
            mlflow.log_params(
                {
                    "min_clusters": min_clusters,
                    "max_clusters": max_clusters,
                    "sample_fraction": sample_fraction,
                    "max_sample_size": max_sample_size,
                    "random_state": 42,
                    "dbscan_min_samples": dbscan_min_samples,
                }
            )
            mlflow.set_tag("pipeline", "Role Mining Clustering")

            if df is not None and not df.empty:
                # Transform to binary access matrix
                binary_access_matrix = transform_to_binary_matrix(df)

                # Prepare TF-IDF-transformed data for K-Means and Agglomerative Clustering
                tfidf_transformer = TfidfTransformer()
                tfidf_matrix = tfidf_transformer.fit_transform(binary_access_matrix)
                clustering_data = tfidf_matrix.toarray()

                # Prepare binary data for DBSCAN
                dbscan_data = binary_access_matrix.values

                # Find the optimal number of clusters using TF-IDF-transformed data
                optimal_cluster_count = find_optimal_clusters(
                    clustering_data, min_clusters, max_clusters
                )

                # K-distance plot to help determine eps for DBSCAN using binary data
                k_distances = plot_k_distance(dbscan_data, dbscan_min_samples)

                # Automatically detect the elbow point to estimate eps
                if dbscan_eps is None:
                    eps_estimated = detect_eps(k_distances)
                    if eps_estimated is not None:
                        dbscan_eps = eps_estimated
                        mlflow.log_param("dbscan_eps_estimated", dbscan_eps)
                        print(f"Estimated eps value for DBSCAN: {dbscan_eps}")
                    else:
                        print(
                            "Could not automatically estimate eps for DBSCAN. Using default value."
                        )
                        mlflow.log_param("dbscan_eps_estimated", "None")
                else:
                    mlflow.log_param("dbscan_eps_provided", dbscan_eps)

                # Perform K-Means and Agglomerative Clustering using TF-IDF-transformed data
                (
                    kmeans_labels,
                    hierarchical_labels,
                ) = perform_kmeans_and_hierarchical(
                    clustering_data, optimal_cluster_count
                )

                # Perform DBSCAN clustering using binary data
                dbscan_labels = perform_dbscan(
                    dbscan_data, dbscan_eps, dbscan_min_samples
                )

                # Analyze K-Means clusters using original binary data
                binary_access_matrix["k_means_clusters"] = kmeans_labels
                analyze_clusters(binary_access_matrix)

                # Analyze hierarchical clusters
                binary_access_matrix["hierarchical_cluster"] = hierarchical_labels
                analyze_hierarchical_clusters(binary_access_matrix, hierarchical_labels)

                # Analyze DBSCAN clusters
                binary_access_matrix["dbscan_cluster"] = dbscan_labels
                analyze_dbscan_clusters(binary_access_matrix, dbscan_labels)

                # Get the list of feature names used during fit
                cluster_label_columns = [
                    "k_means_clusters",
                    "hierarchical_cluster",
                    "dbscan_cluster",
                ]
                feature_names = binary_access_matrix.columns.difference(
                    cluster_label_columns
                )

                # Sampling for dendrogram visualization using hierarchical clusters
                subset_data = (
                    binary_access_matrix.groupby(
                        "hierarchical_cluster", group_keys=False
                    )
                    .apply(
                        lambda x: (
                            x.sample(frac=sample_fraction, random_state=42)
                            if len(x) * sample_fraction <= max_sample_size
                            else x.sample(n=max_sample_size, random_state=42)
                        )
                    )
                    .reset_index(drop=True)
                )

                # Select only the original feature columns
                subset_data = subset_data[feature_names]

                # Transform subset data for plotting (using TF-IDF)
                subset_tfidf = tfidf_transformer.transform(subset_data)
                subset_tfidf_dense = subset_tfidf.toarray()

                # Plot dendrogram using the subset of transformed data
                plot_dendrogram(subset_tfidf_dense, subset_data.index.tolist())

    except Exception as e:
        logger.error("An error occurred in run_pipeline:", exc_info=True)
        mlflow.log_param("run_pipeline_error", str(e))
        raise
