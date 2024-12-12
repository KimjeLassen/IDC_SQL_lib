import pytest
from unittest.mock import MagicMock, patch
from app.clustering.clustering_pipeline import run_pipeline
from app.database.models import ClusteringRun
from sqlalchemy.orm import Session
import pandas as pd

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    db = MagicMock(spec=Session)
    return db

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    data = {
        "user_id": [1, 2, 3],
        "role_1": [1, 0, 1],
        "role_2": [0, 1, 0],
        "role_3": [1, 1, 1],
    }
    df = pd.DataFrame(data).set_index("user_id")
    return df

@pytest.fixture
def mock_clustering_run():
    """Create a mock ClusteringRun object."""
    return ClusteringRun(run_id="test_run", status="running", algorithm="kmeans")

@patch("app.clustering.clustering_pipeline.get_or_create_clustering_run")
@patch("app.clustering.clustering_pipeline.transform_to_binary_matrix")
@patch("app.clustering.clustering_pipeline.select_and_run_algorithm")
@patch("app.clustering.clustering_pipeline.extract_cluster_details")
@patch("app.clustering.clustering_pipeline.update_clustering_run_success")
@patch("app.clustering.clustering_pipeline.update_clustering_run_failure")
def test_run_pipeline_success(
    mock_update_failure,
    mock_update_success,
    mock_extract_details,
    mock_select_algorithm,
    mock_transform_matrix,
    mock_get_or_create,
    mock_db_session,
    sample_dataframe,
    mock_clustering_run,
):
    """Test run_pipeline for successful execution."""
    # Mock behavior of imported functions
    mock_get_or_create.return_value = mock_clustering_run
    mock_transform_matrix.return_value = sample_dataframe
    mock_select_algorithm.return_value = sample_dataframe
    mock_extract_details.return_value = [{"cluster_label": "0", "user_ids": [1, 3]}]

    # Call the run_pipeline function
    run_pipeline(
        db=mock_db_session,
        df=sample_dataframe,
        algorithm="kmeans",
        n_clusters=2,
        run_id="test_run",
    )

    # Assertions
    mock_get_or_create.assert_called_once_with(mock_db_session, "test_run", "kmeans")
    mock_transform_matrix.assert_called_once_with(sample_dataframe)
    mock_select_algorithm.assert_called_once()
    mock_extract_details.assert_called_once()
    mock_update_success.assert_called_once()
    mock_update_failure.assert_not_called()


@patch("app.clustering.clustering_pipeline.get_or_create_clustering_run")
@patch("app.clustering.clustering_pipeline.transform_to_binary_matrix")
@patch("app.clustering.clustering_pipeline.update_clustering_run_failure")
def test_run_pipeline_failure(
    mock_update_failure,
    mock_transform_matrix,
    mock_get_or_create,
    mock_db_session,
    sample_dataframe,
    mock_clustering_run,
):
    """Test run_pipeline for failure handling."""
    # Mock behavior of imported functions
    mock_get_or_create.return_value = mock_clustering_run
    mock_transform_matrix.side_effect = Exception("Data transformation failed")

    # Call the run_pipeline function and expect an exception
    with pytest.raises(Exception, match="Data transformation failed"):
        run_pipeline(
            db=mock_db_session,
            df=sample_dataframe,
            algorithm="kmeans",
            run_id="test_run",
        )

    # Assertions
    mock_get_or_create.assert_called_once_with(mock_db_session, "test_run", "kmeans")
    mock_transform_matrix.assert_called_once_with(sample_dataframe)
    mock_update_failure.assert_called_once()
