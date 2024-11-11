import unittest
import numpy as np
import pandas as pd
from clustering.dbscan.dbscan_cluster import (
    analyze_dbscan_clusters,
    plot_k_distance,
    detect_eps,
)
from unittest.mock import patch


class TestDBSCANCluster(unittest.TestCase):
    """
    Unit tests for the DBSCAN clustering functions in the `dbscan_cluster` module.
    This test suite verifies the behavior of distance plotting, epsilon detection,
    and cluster analysis for DBSCAN clustering.
    """

    @patch("clustering.dbscan.dbscan_cluster.plt.show")
    def test_plot_k_distance(self, mock_show):
        """
        Test the `plot_k_distance` function to ensure it correctly calculates
        the k-distances and generates a plot without errors.

        Mocks:
        - `plt.show`: Mocked to suppress the plot display during testing.

        Asserts:
        - The length of the returned k-distances matches the number of input data points.
        """
        # Arrange: Generate random input data
        data = np.random.rand(10, 2)
        min_samples = 2

        # Act: Call the function under test
        k_distances = plot_k_distance(data, min_samples)

        # Assert: Verify the length of the k-distances array
        self.assertEqual(len(k_distances), len(data))

    def test_detect_eps(self):
        """
        Test the `detect_eps` function to ensure it correctly identifies
        the elbow point in the k-distance plot and returns a valid epsilon value.

        Asserts:
        - The returned epsilon value is not `None`.
        - The epsilon value falls within the expected range of the k-distances.
        """
        # Arrange: Create k-distances with a noticeable knee point
        k_distances = np.concatenate(
            [np.linspace(0, 0.05, 50), np.linspace(0.15, 0.5, 50)]
        )

        # Act: Call the function under test
        eps = detect_eps(k_distances)

        # Assert: Verify the epsilon value
        self.assertIsNotNone(eps, "detect_eps returned None")
        self.assertTrue(0 < eps < 0.5, f"eps value {eps} not in expected range")

    @patch("clustering.dbscan.dbscan_cluster.mlflow")
    def test_analyze_dbscan_clusters(self, mock_mlflow):
        """
        Test the `analyze_dbscan_clusters` function to ensure it correctly
        assigns DBSCAN cluster labels and analyzes the contents of each cluster.

        Mocks:
        - `mlflow`: Mocked to verify that logging functions are called during analysis.

        Asserts:
        - The output DataFrame includes the 'dbscan_cluster' column with the cluster labels.
        """
        # Arrange: Set up the input DataFrame and DBSCAN labels
        df = pd.DataFrame(
            {"user_id": [1, 2, 3], "role1": [1, 0, 1], "role2": [0, 1, 0]}
        ).set_index("user_id")
        dbscan_labels = np.array([0, 0, -1])

        # Use a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Act: Call the function under test
        analyze_dbscan_clusters(df_copy, dbscan_labels)

        # Assert: Verify that the 'dbscan_cluster' column was added
        self.assertIn("dbscan_cluster", df_copy.columns)


if __name__ == "__main__":
    unittest.main()
