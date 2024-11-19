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


class ClusteringRequest(BaseModel):
    algorithm: Literal["kmeans", "hierarchical", "dbscan"]
    min_clusters: Optional[int] = 3
    max_clusters: Optional[int] = 8
    dbscan_eps: Optional[float] = None
    dbscan_min_samples: Optional[int] = 5

    @model_validator(mode="before")
    def check_parameters(cls, values):
        algorithm = values.get("algorithm")

        if algorithm != "dbscan":
            values["dbscan_eps"] = None
            values["dbscan_min_samples"] = None
        else:
            dbscan_eps = values.get("dbscan_eps")
            if dbscan_eps is None or dbscan_eps <= 0:
                raise ValueError(
                    "`dbscan_eps` must be a positive float for DBSCAN algorithm."
                )

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

    # Run the clustering pipeline asynchronously
    background_tasks.add_task(
        execute_clustering,
        df=df,
        algorithm=request.algorithm,
        min_clusters=request.min_clusters,
        max_clusters=request.max_clusters,
        dbscan_eps=request.dbscan_eps,
        dbscan_min_samples=request.dbscan_min_samples,
        run_id=run_id,
    )

    return ClusteringResponse(run_id=run_id, message="Clustering run initiated.")


def execute_clustering(
    df, algorithm, min_clusters, max_clusters, dbscan_eps, dbscan_min_samples, run_id
):
    """
    Wrapper function to execute the clustering pipeline and store results.
    """
    try:
        RUN_STATUS[run_id] = "running"
        run_pipeline(
            df=df,
            algorithm=algorithm,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            run_id=run_id,
        )
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
