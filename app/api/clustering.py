# app/api/clustering.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, model_validator
from typing import List, Optional, Literal
import uuid
import logging
from app.database.db_base import get_db
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.database.models import ClusteringRun
from app.database.db_base import SessionLocal
from app.clustering.enums.enums import (
    HierarchicalLinkage,
    HierarchicalMetric,
    DBSCANMetric,
    DBSCANAlgorithm,
)

from app.database.fetch_data import fetch_data
from app.clustering.clustering_pipeline import (
    run_pipeline,
    get_available_algorithms,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class ClusteringRequest(BaseModel):
    algorithm: Literal["kmeans", "hierarchical", "dbscan"]

    # Common parameters
    random_state: int = 42

    # Parameters for K-Means and Hierarchical
    n_clusters: Optional[int] = None
    min_clusters: int = 3
    max_clusters: int = 8

    # Parameters specific to K-Means
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300

    # Parameters specific to Hierarchical Clustering
    hierarchical_linkage: HierarchicalLinkage = HierarchicalLinkage.ward
    hierarchical_metric: HierarchicalMetric = HierarchicalMetric.euclidean

    # Parameters specific to DBSCAN
    dbscan_eps: Optional[float] = None
    dbscan_min_samples: int = 5
    dbscan_metric: DBSCANMetric = DBSCANMetric.euclidean
    dbscan_algorithm: DBSCANAlgorithm = DBSCANAlgorithm.auto

    @model_validator(mode="after")
    def check_parameters(cls, model):
        algorithm = model.algorithm

        if algorithm == "kmeans":
            # K-Means specific validations
            n_clusters = model.n_clusters
            min_clusters = model.min_clusters
            max_clusters = model.max_clusters
            if n_clusters is not None and n_clusters <= 0:
                raise ValueError("`n_clusters` must be a positive integer.")
            if min_clusters <= 0 or max_clusters <= 0 or min_clusters > max_clusters:
                raise ValueError(
                    "`min_clusters` and `max_clusters` must be positive integers, and `min_clusters` <= `max_clusters`."
                )

        elif algorithm == "hierarchical":
            linkage = model.hierarchical_linkage
            metric = model.hierarchical_metric

            if (
                linkage == HierarchicalLinkage.ward
                and metric != HierarchicalMetric.euclidean
            ):
                raise ValueError(
                    "When `hierarchical_linkage` is 'ward', `hierarchical_metric` must be 'euclidean'."
                )

            # Validate cluster numbers as in K-Means
            n_clusters = model.n_clusters
            min_clusters = model.min_clusters
            max_clusters = model.max_clusters
            if n_clusters is not None and n_clusters <= 0:
                raise ValueError("`n_clusters` must be a positive integer.")
            if min_clusters <= 0 or max_clusters <= 0 or min_clusters > max_clusters:
                raise ValueError(
                    "`min_clusters` and `max_clusters` must be positive integers, and `min_clusters` <= `max_clusters`."
                )

        elif algorithm == "dbscan":
            dbscan_eps = model.dbscan_eps
            dbscan_min_samples = model.dbscan_min_samples

            if dbscan_eps is not None and dbscan_eps <= 0:
                raise ValueError("`dbscan_eps` must be a positive float.")
            if dbscan_min_samples <= 0:
                raise ValueError("`dbscan_min_samples` must be a positive integer.")

        return model


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


@router.get("/parameters")
def get_algorithm_parameters():
    """
    Get the possible values for clustering algorithm parameters.
    """
    return {
        "hierarchical_linkage_options": [e.value for e in HierarchicalLinkage],
        "hierarchical_metric_options": [e.value for e in HierarchicalMetric],
        "dbscan_metric_options": [e.value for e in DBSCANMetric],
        "dbscan_algorithm_options": [e.value for e in DBSCANAlgorithm],
    }


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

    # Create ClusteringRun record with status 'pending'
    clustering_run = ClusteringRun(
        run_id=run_id, status="pending", algorithm=request.algorithm
    )
    db.add(clustering_run)
    db.commit()

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
    db = SessionLocal()
    try:
        # Update the run status in the database to 'running'
        clustering_run = db.query(ClusteringRun).filter_by(run_id=run_id).first()
        if clustering_run:
            clustering_run.status = "running"
            clustering_run.started_at = datetime.now(timezone.utc)
            db.commit()
        else:
            # Should not happen, but handle it
            clustering_run = ClusteringRun(
                run_id=run_id,
                status="running",
                algorithm=kwargs.get("algorithm"),
                started_at=datetime.now(timezone.utc),
            )
            db.add(clustering_run)
            db.commit()

        # Call run_pipeline
        run_pipeline(db=db, **kwargs)
    except Exception as e:
        logger.error(f"Clustering run {run_id} failed: {e}")
        # Update the run status in the database
        clustering_run = db.query(ClusteringRun).filter_by(run_id=run_id).first()
        if clustering_run:
            clustering_run.status = "failed"
            clustering_run.finished_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


@router.get("/results/{run_id}", response_model=ClusteringResultResponse)
def get_results(run_id: str, db: Session = Depends(get_db)):
    """
    Retrieve clustering results for a given run ID.
    """
    clustering_run = db.query(ClusteringRun).filter_by(run_id=run_id).first()
    if not clustering_run:
        raise HTTPException(status_code=404, detail="Run ID not found.")

    status = clustering_run.status
    results = clustering_run.results

    return ClusteringResultResponse(clusters=results, run_id=run_id, status=status)
