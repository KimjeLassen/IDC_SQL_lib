# app/clustering/clustering_pipeline.py

import logging
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from app.clustering.kmeans.k_means_cluster import (
    analyze_clusters,
    find_optimal_clusters,
)
from app.clustering.dbscan.dbscan_cluster import (
    analyze_dbscan_clusters,
    calculate_k_distance,
    detect_eps,
)
from app.clustering.hierarchical.hierarchical_cluster import (
    analyze_hierarchical_clusters,
)
from app.clustering.data_manipulation.data_manipulation import (
    transform_to_binary_matrix,
    perform_dbscan,
    perform_kmeans_and_hierarchical,
)
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionaries to store clustering results and statuses
CLUSTERING_RESULTS = {}
RUN_STATUS = {}


def safe_start_run(run_name="Clustering_Run"):
    """
    Safely start an MLflow run by ending any active runs first.
    """
    mlflow.end_run()  # Ends any active run
    mlflow.start_run(run_name=run_name)
    return mlflow.active_run()


def get_available_algorithms():
    """
    Return a list of available clustering algorithms.
    """
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
    """
    Execute the clustering pipeline based on the selected algorithm and parameters.
    Stores the results in CLUSTERING_RESULTS dictionary and updates RUN_STATUS.
    """

    if run_id is None:
        run_id = str(uuid.uuid4())

    # Validate input DataFrame
    required_columns = {"user_id", "system_role_name"}
    if not required_columns.issubset(df.columns):
        error_msg = f"Input DataFrame must contain columns: {required_columns}"
        logger.error(error_msg)
        mlflow.log_param("input_validation_error", error_msg)
        RUN_STATUS[run_id] = "failed"
        raise ValueError(error_msg)

    # Initialize status as running
    RUN_STATUS[run_id] = "running"

    mlflow.set_experiment("Role Mining Clustering Experiment")
    try:
        with safe_start_run(run_name="Clustering_Run"):
            logger.info("Starting clustering pipeline...")

            # Log pipeline parameters
            params = {
                "clustering_algorithm": algorithm,
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
            mlflow.set_tag("pipeline", "Role Mining Clustering")

            if df is not None and not df.empty:
                # Transform to binary access matrix
                logger.info("Transforming data to binary access matrix...")

                binary_access_matrix = transform_to_binary_matrix(df)

                logger.info("Data transformation complete.")

                if algorithm in ["kmeans", "hierarchical"]:
                    logger.info(
                        f"Preparing TF-IDF transformation for {algorithm} clustering..."
                    )

                    # Prepare TF-IDF-transformed data
                    tfidf_transformer = TfidfTransformer()
                    tfidf_matrix = tfidf_transformer.fit_transform(binary_access_matrix)
                    clustering_data = tfidf_matrix.toarray()

                    logger.info("TF-IDF transformation complete.")

                    # Find the optimal number of clusters using silhouette score
                    logger.info(
                        f"Finding the optimal number of clusters for {algorithm}..."
                    )

                    optimal_cluster_count = find_optimal_clusters(
                        clustering_data, min_clusters, max_clusters
                    )

                    logger.info(f"Optimal number of clusters: {optimal_cluster_count}")

                    # Perform K-Means and/or Hierarchical Clustering
                    if algorithm == "kmeans":
                        logger.info("Performing K-Means clustering...")

                        kmeans_labels, _ = perform_kmeans_and_hierarchical(
                            clustering_data, optimal_cluster_count
                        )
                        # Analyze K-Means clusters using original binary data
                        binary_access_matrix["k_means_clusters"] = kmeans_labels

                        logger.info("K-Means clustering complete.")
                        logger.info("Analyzing K-Means clusters...")

                        analyze_clusters(binary_access_matrix)

                        logger.info("K-Means cluster analysis complete.")

                    elif algorithm == "hierarchical":
                        logger.info("Performing Hierarchical clustering...")

                        _, hierarchical_labels = perform_kmeans_and_hierarchical(
                            clustering_data, optimal_cluster_count
                        )

                        logger.info("Hierarchical clustering complete.")
                        logger.info("Analyzing Hierarchical clusters...")

                        # Analyze hierarchical clusters
                        binary_access_matrix[
                            "hierarchical_cluster"
                        ] = hierarchical_labels
                        analyze_hierarchical_clusters(
                            binary_access_matrix, hierarchical_labels
                        )

                        logger.info("Hierarchical cluster analysis complete.")

                elif algorithm == "dbscan":
                    logger.info("Preparing data for DBSCAN clustering...")

                    # Prepare binary data for DBSCAN
                    dbscan_data = binary_access_matrix.values

                    # K-distance calculation to help determine eps for DBSCAN using binary data
                    k_distances = calculate_k_distance(dbscan_data, dbscan_min_samples)

                    # Automatically detect the elbow point to estimate eps
                    if dbscan_eps is None:
                        eps_estimated = detect_eps(k_distances)
                        if eps_estimated is not None:
                            dbscan_eps = eps_estimated
                            mlflow.log_param("dbscan_eps_estimated", dbscan_eps)
                            logger.info(f"Estimated eps value for DBSCAN: {dbscan_eps}")
                        else:
                            logger.info(
                                "Could not automatically estimate eps for DBSCAN. Using default value."
                            )
                            mlflow.log_param("dbscan_eps_estimated", "None")
                    else:
                        mlflow.log_param("dbscan_eps_provided", dbscan_eps)

                    logger.info("Performing DBSCAN clustering...")

                    # Perform DBSCAN clustering using binary data
                    dbscan_labels = perform_dbscan(
                        dbscan_data, dbscan_eps, dbscan_min_samples
                    )
                    binary_access_matrix["dbscan_cluster"] = dbscan_labels

                    logger.info("DBSCAN clustering complete.")
                    logger.info("Analyzing DBSCAN clusters...")

                    analyze_dbscan_clusters(binary_access_matrix, dbscan_labels)

                    logger.info("DBSCAN cluster analysis complete.")

                # Store results

                logger.info("Extracting cluster details...")

                CLUSTERING_RESULTS[run_id] = extract_cluster_details(
                    binary_access_matrix, algorithm
                )

                logger.info("Cluster details extracted and stored.")

                # Update status as completed
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
        if algorithm == "dbscan" and cluster_label == -1:
            label = "Noise"
        else:
            label = str(int(cluster_label)) if cluster_label != -1 else "Noise"

        user_ids = cluster_data.index.tolist()
        role_columns = binary_access_matrix.columns.difference([label_column])
        role_details = cluster_data[role_columns].sum().to_dict()

        cluster_details.append(
            {"cluster_label": label, "user_ids": user_ids, "role_details": role_details}
        )

    return cluster_details


def get_clustering_results(run_id):
    """
    Retrieve clustering results for a given run_id.
    """
    return CLUSTERING_RESULTS.get(run_id, [])
