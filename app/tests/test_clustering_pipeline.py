import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from clustering.clustering_pipeline import run_pipeline


class TestClusteringPipeline(unittest.TestCase):
    """
    Unit tests for the `run_pipeline` function in the clustering pipeline module.
    This test suite ensures that the main pipeline function executes correctly and
    calls all the expected components with the correct inputs and outputs.
    """

    @patch("clustering.clustering_pipeline.mlflow")
    @patch("clustering.clustering_pipeline.transform_to_binary_matrix")
    @patch("clustering.clustering_pipeline.TfidfTransformer")
    @patch("clustering.clustering_pipeline.find_optimal_clusters")
    @patch("clustering.clustering_pipeline.plot_k_distance")
    @patch("clustering.clustering_pipeline.detect_eps")
    @patch("clustering.clustering_pipeline.perform_kmeans_and_hierarchical")
    @patch("clustering.clustering_pipeline.perform_dbscan")
    @patch("clustering.clustering_pipeline.analyze_clusters")
    @patch("clustering.clustering_pipeline.analyze_hierarchical_clusters")
    @patch("clustering.clustering_pipeline.analyze_dbscan_clusters")
    @patch("clustering.clustering_pipeline.plot_dendrogram")
    def test_run_pipeline(
        self,
        mock_plot_dendrogram,
        mock_analyze_dbscan_clusters,
        mock_analyze_hierarchical_clusters,
        mock_analyze_clusters,
        mock_perform_dbscan,
        mock_perform_kmeans_and_hierarchical,
        mock_detect_eps,
        mock_find_optimal_clusters,
        mock_tfidf_transformer,
        mock_transform_to_binary_matrix,
    ):
        """
        Test the `run_pipeline` function to ensure it executes the entire
        clustering process and calls each step of the pipeline correctly.

        Mocks are used for all external dependencies to isolate the function
        being tested and focus on verifying its logic and control flow.
        """

        # Arrange: Set up the test data and mock return values

        # Input DataFrame with user-role mappings
        df = pd.DataFrame(
            {"user_id": [1, 2, 3], "system_role_name": ["role1", "role2", "role3"]}
        )

        # Mock the binary access matrix after transforming user-role mappings
        binary_matrix = pd.DataFrame(
            {"role1": [1, 0, 0], "role2": [0, 1, 0], "role3": [0, 0, 1]},
            index=[1, 2, 3],
        )
        mock_transform_to_binary_matrix.return_value = binary_matrix

        # Mock return values for clustering functions
        mock_find_optimal_clusters.return_value = 2
        mock_detect_eps.return_value = 0.5

        # Mock TF-IDF transformation
        mock_tfidf_instance = MagicMock()
        mock_tfidf_instance.fit_transform.return_value = MagicMock(
            toarray=lambda: [[0, 1], [1, 0], [0, 1]]
        )
        mock_tfidf_transformer.return_value = mock_tfidf_instance

        # Create mock cluster labels for K-Means, hierarchical, and DBSCAN
        kmeans_labels = [0, 1, 0]
        hierarchical_labels = [1, 0, 1]
        dbscan_labels = [0, -1, 0]

        # Set mock return values for clustering analysis functions
        mock_perform_kmeans_and_hierarchical.return_value = (
            kmeans_labels,
            hierarchical_labels,
        )
        mock_perform_dbscan.return_value = dbscan_labels

        # Act: Call the function under test
        run_pipeline(df)

        # Assert: Verify that all mocked functions were called as expected

        # Ensure the binary transformation function was called once with the input DataFrame
        mock_transform_to_binary_matrix.assert_called_once_with(df)

        # Check if the optimal cluster count was determined
        mock_find_optimal_clusters.assert_called()

        # Verify that K-Means and hierarchical clustering were performed
        mock_perform_kmeans_and_hierarchical.assert_called()

        # Confirm that DBSCAN clustering was performed
        mock_perform_dbscan.assert_called()

        # Validate that cluster analysis functions were executed
        mock_analyze_clusters.assert_called()
        mock_analyze_hierarchical_clusters.assert_called()
        mock_analyze_dbscan_clusters.assert_called()

        # Ensure the dendrogram plot was generated
        mock_plot_dendrogram.assert_called()


if __name__ == "__main__":
    unittest.main()
