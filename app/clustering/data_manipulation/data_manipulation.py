# app/clustering/data_manipulation/data_manipulation.py
import pandas as pd
import joblib
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_k_distance(data, min_samples):
    """
    Calculate k-distances without generating a plot.
    """
    try:
        data_dense = data.toarray() if hasattr(data, "toarray") else data

        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(data_dense)
        distances, indices = neighbors_fit.kneighbors(data_dense)
        k_distances = np.sort(distances[:, min_samples - 1])

        return k_distances
    except Exception:
        logger.error("An error occurred while calculating k-distances:", exc_info=True)
        raise


def detect_eps(k_distances):
    """
    Estimate the optimal `eps` value for DBSCAN by detecting the "elbow" point in the k-distance plot.
    """
    indices = np.arange(len(k_distances))
    kneedle = KneeLocator(
        indices, k_distances, S=1.0, curve="convex", direction="increasing"
    )
    elbow_index = kneedle.knee
    if elbow_index is not None:
        eps_estimated = k_distances[elbow_index]
        return eps_estimated
    else:
        return None


def find_optimal_clusters(
    data, min_clusters, max_clusters, algorithm="kmeans", **kwargs
):
    highest_silhouette_score = -1
    optimal_n_clusters = min_clusters

    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            if algorithm == "kmeans":
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=kwargs.get("random_state", 42),
                    n_init=kwargs.get("n_init", 10),
                    max_iter=kwargs.get("max_iter", 300),
                )
            elif algorithm == "hierarchical":
                linkage = kwargs.get("linkage", "ward")
                metric = kwargs.get("metric", "euclidean")
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    metric=metric,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            cluster_labels = model.fit_predict(data)

            # Calculate the silhouette score for the current number of clusters
            silhouette_avg = silhouette_score(data, cluster_labels)

            # Update the optimal number of clusters if the score improves
            if silhouette_avg > highest_silhouette_score:
                highest_silhouette_score = silhouette_avg
                optimal_n_clusters = n_clusters
        except Exception:
            # Log any errors encountered during the process
            logger.error(
                f"An error occurred while finding optimal clusters for n_clusters={n_clusters} with algorithm={algorithm}:",
                exc_info=True,
            )

    # Log the optimal number of clusters and the highest silhouette score
    logger.info(
        f"The optimal number of clusters is: {optimal_n_clusters} with a {algorithm} silhouette score of {highest_silhouette_score}"
    )

    return optimal_n_clusters


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

    """
    # Convert 'system_role_name' to dummy variables (one-hot encoding)
    binary_access_matrix = pd.get_dummies(
        df, columns=["system_role_name"], prefix="", prefix_sep=""
    )
    # Group by 'user_id' and take the maximum to aggregate multiple roles per user
    binary_access_matrix = binary_access_matrix.groupby("user_id").max()
    # Log the number of unique users and roles
    return binary_access_matrix


def perform_kmeans(data, n_clusters, random_state=42, n_init=10, max_iter=300):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    kmeans_labels = kmeans.fit_predict(data)
    return kmeans_labels


def perform_hierarchical(data, n_clusters, linkage="ward", metric="euclidean"):
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage, metric=metric
    )
    hierarchical_labels = hierarchical.fit_predict(data)
    return hierarchical_labels


def perform_dbscan(
    data, dbscan_eps, dbscan_min_samples, metric="euclidean", algorithm="auto"
):
    """
    Apply DBSCAN clustering to data, estimate silhouette scores if applicable, and log the results.
    """
    try:
        # DBSCAN clustering
        dbscan = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            metric=metric,
            algorithm=algorithm,
        )
        dbscan_labels = dbscan.fit_predict(data)

        # Calculate and log the silhouette score if there are sufficient clusters
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        if n_clusters > 1:
            core_samples_mask = dbscan_labels != -1
            dbscan_silhouette = silhouette_score(
                data[core_samples_mask], dbscan_labels[core_samples_mask]
            )
            logger.info(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
        else:
            logger.info(
                "DBSCAN did not find enough clusters to compute silhouette score."
            )

        # Save the DBSCAN model as an artifact
        dbscan_model_path = "dbscan_model.pkl"
        joblib.dump(dbscan, dbscan_model_path)

        return dbscan_labels
    except Exception:
        # Log error and traceback if clustering fails
        logger.error("An error occurred during DBSCAN clustering:", exc_info=True)
        raise
