# app/clustering/clustering_pipeline
import logging
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
from app.database.models import ClusteringRun
from datetime import datetime, timezone
from app.clustering.enums.enums import (
    HierarchicalLinkage,
    HierarchicalMetric,
    DBSCANMetric,
    DBSCANAlgorithm,
)
from sqlalchemy.orm import Session


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_algorithms():
    """Return a list of available clustering algorithms."""
    return ["kmeans", "hierarchical", "dbscan"]


def run_pipeline(
    db: Session,
    df,
    algorithm,
    n_clusters=None,
    min_clusters=3,
    max_clusters=8,
    dbscan_eps=None,
    dbscan_min_samples=5,
    random_state=42,
    kmeans_n_init=10,
    kmeans_max_iter=300,
    hierarchical_linkage: HierarchicalLinkage = HierarchicalLinkage.ward,
    hierarchical_metric: HierarchicalMetric = HierarchicalMetric.euclidean,
    dbscan_metric: DBSCANMetric = DBSCANMetric.euclidean,
    dbscan_algorithm: DBSCANAlgorithm = DBSCANAlgorithm.auto,
    run_id=None,
):
    """Execute the clustering pipeline and store results."""

    if run_id is None:
        run_id = str(uuid.uuid4())

    # Retrieve the ClusteringRun record
    clustering_run = get_or_create_clustering_run(db, run_id, algorithm)

    try:
        logger.info("Starting clustering pipeline...")

        # Prepare data
        binary_access_matrix = transform_to_binary_matrix(df)

        # Select and run the appropriate clustering algorithm
        clustering_results = select_and_run_algorithm(
            algorithm,
            binary_access_matrix,
            n_clusters,
            min_clusters,
            max_clusters,
            random_state,
            kmeans_n_init,
            kmeans_max_iter,
            hierarchical_linkage,
            hierarchical_metric,
            dbscan_eps,
            dbscan_min_samples,
            dbscan_metric,
            dbscan_algorithm,
            clustering_run,
        )

        # Extract and store clustering results
        results = extract_cluster_details(clustering_results, algorithm)

        # Update the ClusteringRun record with results and status
        update_clustering_run_success(db, clustering_run, results, run_id)

    except Exception:
        logger.error("An error occurred in run_pipeline:", exc_info=True)
        update_clustering_run_failure(db, clustering_run)
        raise


def get_or_create_clustering_run(
    db: Session, run_id: str, algorithm: str
) -> ClusteringRun:
    """Retrieve or create a ClusteringRun record."""
    clustering_run = db.query(ClusteringRun).filter_by(run_id=run_id).first()
    if not clustering_run:
        clustering_run = ClusteringRun(
            run_id=run_id,
            status="running",
            algorithm=algorithm,
            started_at=datetime.now(timezone.utc),
        )
        db.add(clustering_run)
        db.commit()
    else:
        # Update status to 'running'
        clustering_run.status = "running"
        clustering_run.started_at = datetime.now(timezone.utc)
        db.commit()
    return clustering_run


def select_and_run_algorithm(
    algorithm: str,
    binary_access_matrix,
    n_clusters,
    min_clusters,
    max_clusters,
    random_state,
    kmeans_n_init,
    kmeans_max_iter,
    hierarchical_linkage,
    hierarchical_metric,
    dbscan_eps,
    dbscan_min_samples,
    dbscan_metric,
    dbscan_algorithm,
    clustering_run: ClusteringRun,
):
    """Select and run the appropriate clustering algorithm."""
    if algorithm in ["kmeans", "hierarchical"]:
        # Apply TF-IDF transformation
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(binary_access_matrix)
        clustering_data = tfidf_matrix.toarray()

        # Determine the number of clusters
        if algorithm == "kmeans":
            optimal_cluster_count = determine_cluster_count(
                clustering_data,
                n_clusters,
                min_clusters,
                max_clusters,
                algorithm,
                random_state=random_state,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
            )
        elif algorithm == "hierarchical":
            optimal_cluster_count = determine_cluster_count(
                clustering_data,
                n_clusters,
                min_clusters,
                max_clusters,
                algorithm,
                linkage=hierarchical_linkage.value,
                metric=hierarchical_metric.value,
            )

        logger.info(f"Optimal number of clusters: {optimal_cluster_count}")

        if algorithm == "kmeans":
            labels = run_kmeans_clustering(
                clustering_data,
                optimal_cluster_count,
                random_state,
                kmeans_n_init,
                kmeans_max_iter,
            )
            binary_access_matrix["cluster_label"] = labels
        else:
            labels = run_hierarchical_clustering(
                clustering_data,
                optimal_cluster_count,
                hierarchical_linkage,
                hierarchical_metric,
            )
            binary_access_matrix["cluster_label"] = labels

    elif algorithm == "dbscan":
        labels = run_dbscan_clustering(
            binary_access_matrix,
            dbscan_eps,
            dbscan_min_samples,
            dbscan_metric,
            dbscan_algorithm,
        )
        binary_access_matrix["cluster_label"] = labels

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return binary_access_matrix


def determine_cluster_count(
    data, n_clusters, min_clusters, max_clusters, algorithm, **kwargs
):
    """Determine the optimal number of clusters."""
    if n_clusters is not None:
        return n_clusters
    else:
        # Determine the optimal number of clusters
        optimal_cluster_count = find_optimal_clusters(
            data, min_clusters, max_clusters, algorithm=algorithm, **kwargs
        )
        return optimal_cluster_count


def run_kmeans_clustering(
    data,
    n_clusters,
    random_state,
    n_init,
    max_iter,
):
    """Run K-Means clustering."""
    logger.info("Running K-Means clustering...")
    labels = perform_kmeans(
        data,
        n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    return labels


def run_hierarchical_clustering(
    data,
    n_clusters,
    linkage,
    metric,
):
    """Run Hierarchical clustering."""
    logger.info("Running Hierarchical clustering...")
    labels = perform_hierarchical(
        data,
        n_clusters,
        linkage=linkage.value,
        metric=metric.value,
    )
    return labels


def run_dbscan_clustering(
    binary_access_matrix,
    dbscan_eps,
    dbscan_min_samples,
    dbscan_metric,
    dbscan_algorithm,
):
    """Run DBSCAN clustering."""
    logger.info("Running DBSCAN clustering...")
    data = binary_access_matrix.values

    # If dbscan_eps is not provided, estimate it
    if dbscan_eps is None:
        k_distances = calculate_k_distance(data, dbscan_min_samples)
        dbscan_eps = detect_eps(k_distances)
        if dbscan_eps is None:
            error_msg = "Failed to estimate dbscan_eps."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Estimated dbscan_eps: {dbscan_eps}")

    labels = perform_dbscan(
        data,
        dbscan_eps,
        dbscan_min_samples,
        metric=dbscan_metric.value,
        algorithm=dbscan_algorithm.value,
    )
    return labels


def update_clustering_run_success(
    db: Session, clustering_run: ClusteringRun, results, run_id: str
):
    """Update the ClusteringRun record upon success."""
    clustering_run.results = results
    clustering_run.status = "completed"
    clustering_run.finished_at = datetime.now(timezone.utc)
    db.commit()
    logger.info(f"Clustering run {run_id} completed successfully.")


def update_clustering_run_failure(db: Session, clustering_run: ClusteringRun):
    """Update the ClusteringRun record upon failure."""
    clustering_run.status = "failed"
    clustering_run.finished_at = datetime.now(timezone.utc)
    db.commit()


def extract_cluster_details(binary_access_matrix, algorithm):
    """
    Extract cluster details to be returned by the API.
    """
    label_column = "cluster_label"
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

        # Convert any NumPy data types to native Python types
        role_details = {key: int(value) for key, value in role_details.items()}

        cluster_details.append(
            {"cluster_label": label, "user_ids": user_ids, "role_details": role_details}
        )

    return cluster_details
