# app/api/clustering.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, model_validator
from typing import List, Optional, Literal
import uuid
import logging
from app.database.db_base import get_db
from sqlalchemy.orm import Session


from app.database.fetch_data import fetch_data
from app.clustering.clustering_pipeline import (
    run_pipeline,
    get_available_algorithms,
    get_clustering_results,
    RUN_STATUS,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


# app/api/clustering.py


class ClusteringRequest(BaseModel):
    algorithm: Literal["kmeans", "hierarchical", "dbscan"]

    # Common parameters
    random_state: Optional[int] = 42

    # Parameters for K-Means and Hierarchical
    n_clusters: Optional[int] = None  # User can specify or leave it None
    min_clusters: Optional[int] = 3
    max_clusters: Optional[int] = 8

    # Parameters specific to K-Means
    kmeans_n_init: Optional[int] = 10
    kmeans_max_iter: Optional[int] = 300

    # Parameters specific to Hierarchical Clustering
    hierarchical_linkage: Optional[str] = "ward"
    hierarchical_metric: Optional[str] = "euclidean"

    # Parameters specific to DBSCAN
    dbscan_eps: Optional[float] = None  # User can specify or leave it None
    dbscan_min_samples: Optional[int] = 5
    dbscan_metric: Optional[str] = "euclidean"
    dbscan_algorithm: Optional[str] = "auto"

    @model_validator(mode="before")
    def check_parameters(cls, values):
        algorithm = values.get("algorithm")

        if algorithm == "kmeans":
            # Ensure K-Means specific parameters are valid
            if values.get("n_clusters") is not None:
                if values["n_clusters"] <= 0:
                    raise ValueError("`n_clusters` must be a positive integer.")
            else:
                # If n_clusters is not provided, min_clusters and max_clusters must be valid
                min_clusters = values.get("min_clusters", 3)
                max_clusters = values.get("max_clusters", 8)
                if (
                    min_clusters <= 0
                    or max_clusters <= 0
                    or min_clusters > max_clusters
                ):
                    raise ValueError(
                        "`min_clusters` and `max_clusters` must be positive integers, and `min_clusters` <= `max_clusters`."
                    )

            if values.get("kmeans_n_init") is not None and values["kmeans_n_init"] <= 0:
                raise ValueError("`kmeans_n_init` must be a positive integer.")
            if (
                values.get("kmeans_max_iter") is not None
                and values["kmeans_max_iter"] <= 0
            ):
                raise ValueError("`kmeans_max_iter` must be a positive integer.")

        elif algorithm == "hierarchical":
            # Ensure Hierarchical Clustering parameters are valid
            if values.get("n_clusters") is not None:
                if values["n_clusters"] <= 0:
                    raise ValueError("`n_clusters` must be a positive integer.")
            else:
                # If n_clusters is not provided, min_clusters and max_clusters must be valid
                min_clusters = values.get("min_clusters", 3)
                max_clusters = values.get("max_clusters", 8)
                if (
                    min_clusters <= 0
                    or max_clusters <= 0
                    or min_clusters > max_clusters
                ):
                    raise ValueError(
                        "`min_clusters` and `max_clusters` must be positive integers, and `min_clusters` <= `max_clusters`."
                    )

            linkage_options = ["ward", "complete", "average", "single"]
            if values.get("hierarchical_linkage") not in linkage_options:
                raise ValueError(
                    f"`hierarchical_linkage` must be one of {linkage_options}."
                )
            metric_options = [
                "euclidean",
                "l1",
                "l2",
                "manhattan",
                "cosine",
                "precomputed",
            ]
            if values.get("hierarchical_affinity") not in metric_options:
                raise ValueError(
                    f"`hierarchical_affinity` must be one of {metric_options}."
                )
            # For 'ward' linkage, 'affinity' must be 'euclidean'
            if (
                values["hierarchical_linkage"] == "ward"
                and values["hierarchical_metric"] != "euclidean"
            ):
                raise ValueError(
                    "When `hierarchical_linkage` is 'ward', `hierarchical_metric` must be 'euclidean'."
                )

        elif algorithm == "dbscan":
            # Ensure DBSCAN specific parameters are valid
            if values.get("dbscan_eps") is not None:
                if values["dbscan_eps"] <= 0:
                    raise ValueError("`dbscan_eps` must be a positive float.")
            # dbscan_min_samples must be positive
            if (
                values.get("dbscan_min_samples") is not None
                and values["dbscan_min_samples"] <= 0
            ):
                raise ValueError("`dbscan_min_samples` must be a positive integer.")
            # Validate 'dbscan_metric' and 'dbscan_algorithm' if needed
        else:
            raise ValueError("Invalid algorithm selected.")

        return values


class ClusteringResponse(BaseModel):
    run_id: str
    message: str


class AlgorithmListResponse(BaseModel):
    algorithms: List[str]


class ClusterContent(BaseModel):
    cluster_label: str
    user_ids: List[str]
    role_details: dict


class ClusteringResultResponse(BaseModel):
    clusters: Optional[List[ClusterContent]]
    run_id: str
    status: str


@router.get("/algorithms", response_model=AlgorithmListResponse)
def list_available_algorithms():
    """
    List all available clustering algorithms.
    """
    algorithms = get_available_algorithms()
    return AlgorithmListResponse(algorithms=algorithms)


@router.post("/run", response_model=ClusteringResponse)
def run_clustering(
    request: ClusteringRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Run clustering with the specified algorithm and parameters.
    """
    available_algorithms = get_available_algorithms()
    if request.algorithm not in available_algorithms:
        raise HTTPException(status_code=400, detail="Invalid algorithm selected.")

    # Fetch data using the provided session
    df = fetch_data(db)
    if df is None or df.empty:
        raise HTTPException(
            status_code=500, detail="Failed to fetch data from the database."
        )

    # Generate a unique run ID
    run_id = str(uuid.uuid4())

    # Initialize status as pending
    RUN_STATUS[run_id] = "pending"

    # Prepare additional parameters
    clustering_params = request.model_dump()
    clustering_params["df"] = df
    clustering_params["run_id"] = run_id

    # Run the clustering pipeline asynchronously
    background_tasks.add_task(execute_clustering, **clustering_params)

    return ClusteringResponse(run_id=run_id, message="Clustering run initiated.")


def execute_clustering(**kwargs):
    """
    Wrapper function to execute the clustering pipeline and store results.
    """
    run_id = kwargs.get("run_id")
    try:
        RUN_STATUS[run_id] = "running"
        run_pipeline(**kwargs)
    except Exception as e:
        logger.error(f"Clustering run {run_id} failed: {e}")
        RUN_STATUS[run_id] = "failed"


@router.get("/results/{run_id}", response_model=ClusteringResultResponse)
def get_results(run_id: str):
    """
    Retrieve clustering results for a given run ID.
    """
    if run_id not in RUN_STATUS:
        raise HTTPException(status_code=404, detail="Run ID not found.")

    status = RUN_STATUS[run_id]

    if status == "completed":
        clusters = get_clustering_results(run_id)
        return ClusteringResultResponse(clusters=clusters, run_id=run_id, status=status)
    elif status in ["running", "pending"]:
        return ClusteringResultResponse(clusters=None, run_id=run_id, status=status)
    else:
        return ClusteringResultResponse(clusters=None, run_id=run_id, status=status)
