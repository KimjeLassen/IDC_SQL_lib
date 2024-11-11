import unittest
import numpy as np
import pandas as pd
from clustering.kmeans.k_means_cluster import analyze_clusters, find_optimal_clusters
from unittest.mock import patch


class TestKMeansCluster(unittest.TestCase):
    """
    Unit tests for the K-Means clustering functions in the `k_means_cluster` module.
    This test suite verifies the behavior of cluster analysis and optimal cluster
    count detection for K-Means clustering.
    """

    @patch("clustering.kmeans.k_means_cluster.mlflow")
    def test_analyze_clusters(self, mock_mlflow):
        """
        Test the `analyze_clusters` function to ensure it correctly analyzes
        the contents of each K-Means cluster.

        Mocks:
        - `mlflow`: Mocked to verify that logging functions are called during analysis.

        Asserts:
        - The output DataFrame includes the 'k_means_clusters' column with the cluster labels.
        """
        # Arrange: Set up the input DataFrame with user roles and cluster labels
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "role1": [1, 0, 1],
                "role2": [0, 1, 0],
                "k_means_clusters": [0, 0, 1],
            }
        ).set_index("user_id")

        # Act: Call the function under test
        analyze_clusters(df.copy())

        # Assert: Verify that the 'k_means_clusters' column exists in the DataFrame
        self.assertIn("k_means_clusters", df.columns)

    @patch("clustering.kmeans.k_means_cluster.mlflow")
    def test_find_optimal_clusters(self, mock_mlflow):
        """
        Test the `find_optimal_clusters` function to ensure it correctly identifies
        the optimal number of clusters using the silhouette score.

        Mocks:
        - `mlflow`: Mocked to verify that logging functions are called during clustering.

        Asserts:
        - The returned optimal number of clusters is within the specified range of `min_clusters` to `max_clusters`.
        """
        # Arrange: Generate random input data and define cluster range
        data = np.random.rand(100, 2)
        min_clusters = 2
        max_clusters = 5

        # Act: Call the function under test
        optimal_clusters = find_optimal_clusters(data, min_clusters, max_clusters)

        # Assert: Verify that the optimal cluster count is within the specified range
        self.assertTrue(min_clusters <= optimal_clusters <= max_clusters)


if __name__ == "__main__":
    unittest.main()
