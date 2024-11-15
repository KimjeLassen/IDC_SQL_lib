import logging
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from app.clustering.data_manipulation.data_manipulation import (
    transform_to_binary_matrix,
    perform_kmeans,
    perform_hierarchical,
    perform_dbscan,
    calculate_k_distance,
    detect_eps,
    find_optimal_clusters,
)
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionaries to store clustering results and statuses
CLUSTERING_RESULTS = {}
RUN_STATUS = {}


def safe_start_run(run_name="Clustering_Run"):
    """Safely start an MLflow run by ending any active runs first."""
    mlflow.end_run()
    mlflow.start_run(run_name=run_name)
    return mlflow.active_run()


def get_available_algorithms():
    """Return a list of available clustering algorithms."""
    return ["kmeans", "hierarchical", "dbscan"]


def run_pipeline(
    df,
    algorithm,
    min_clusters=3,
    max_clusters=8,
    sample_fraction=0.1,
    max_sample_size=500,
    dbscan_eps=None,
    dbscan_min_samples=5,
    run_id=None,
):
    """Execute the clustering pipeline and store results."""
    if run_id is None:
        run_id = str(uuid.uuid4())

    required_columns = {"user_id", "system_role_name"}
    if not required_columns.issubset(df.columns):
        error_msg = f"Input DataFrame must contain columns: {required_columns}"
        logger.error(error_msg)
        mlflow.log_param("input_validation_error", error_msg)
        RUN_STATUS[run_id] = "failed"
        raise ValueError(error_msg)

    RUN_STATUS[run_id] = "running"
    mlflow.set_experiment("Role Mining Clustering Experiment")

    try:
        with safe_start_run(run_name="Clustering_Run"):
            logger.info("Starting clustering pipeline...")
            params = {
                "algorithm": algorithm,
                "min_clusters": min_clusters,
                "max_clusters": max_clusters,
                "sample_fraction": sample_fraction,
                "max_sample_size": max_sample_size,
                "random_state": 42,
                "dbscan_min_samples": dbscan_min_samples,
            }
            if algorithm == "dbscan":
                params["dbscan_eps"] = dbscan_eps
            mlflow.log_params(params)

            # Transform data to binary access matrix
            binary_access_matrix = transform_to_binary_matrix(df)

            # Apply TF-IDF transformation for K-Means and Hierarchical Clustering
            if algorithm in ["kmeans", "hierarchical"]:
                tfidf_transformer = TfidfTransformer()
                tfidf_matrix = tfidf_transformer.fit_transform(binary_access_matrix)
                clustering_data = tfidf_matrix.toarray()

                # Determine the optimal number of clusters
                optimal_cluster_count = find_optimal_clusters(
                    clustering_data, min_clusters, max_clusters
                )

                # Run the selected algorithm
                if algorithm == "kmeans":
                    kmeans_labels = perform_kmeans(
                        clustering_data, optimal_cluster_count
                    )
                    binary_access_matrix["k_means_clusters"] = kmeans_labels

                elif algorithm == "hierarchical":
                    hierarchical_labels = perform_hierarchical(
                        clustering_data, optimal_cluster_count
                    )
                    binary_access_matrix["hierarchical_cluster"] = hierarchical_labels

            # Run DBSCAN
            elif algorithm == "dbscan":
                dbscan_data = binary_access_matrix.values
                k_distances = calculate_k_distance(dbscan_data, dbscan_min_samples)

                if dbscan_eps is None:
                    eps_estimated = detect_eps(k_distances)
                    dbscan_eps = eps_estimated if eps_estimated else dbscan_eps
                    mlflow.log_param("dbscan_eps_estimated", dbscan_eps)

                dbscan_labels = perform_dbscan(
                    dbscan_data, dbscan_eps, dbscan_min_samples
                )
                binary_access_matrix["dbscan_cluster"] = dbscan_labels

            # Extract and store clustering results
            CLUSTERING_RESULTS[run_id] = extract_cluster_details(
                binary_access_matrix, algorithm
            )
            RUN_STATUS[run_id] = "completed"
            logger.info(f"Clustering run {run_id} completed successfully.")

    except Exception as e:
        logger.error("An error occurred in run_pipeline:", exc_info=True)
        mlflow.log_param("run_pipeline_error", str(e))
        RUN_STATUS[run_id] = "failed"
        raise


def extract_cluster_details(binary_access_matrix, algorithm):
    """
    Extract cluster details to be returned by the API.
    """
    # Define the cluster label column based on the algorithm
    cluster_label_columns = {
        "kmeans": "k_means_clusters",
        "hierarchical": "hierarchical_cluster",
        "dbscan": "dbscan_cluster",
    }

    label_column = cluster_label_columns.get(algorithm)
    if label_column not in binary_access_matrix.columns:
        logger.error(
            f"Cluster label column '{label_column}' not found in binary_access_matrix."
        )
        return []

    clusters = binary_access_matrix.groupby(label_column)
    cluster_details = []

    for cluster_label, cluster_data in clusters:
        # Handle noise points for DBSCAN
        label = (
            "Noise"
            if algorithm == "dbscan" and cluster_label == -1
            else str(int(cluster_label))
        )

        user_ids = cluster_data.index.tolist()
        role_columns = binary_access_matrix.columns.difference([label_column])
        role_details = cluster_data[role_columns].sum().to_dict()

        cluster_details.append(
            {"cluster_label": label, "user_ids": user_ids, "role_details": role_details}
        )

    # Removed dendrogram generation for hierarchical clustering
    return cluster_details


def get_clustering_results(run_id):
    """Retrieve clustering results for a given run_id."""
    return CLUSTERING_RESULTS.get(run_id, [])
