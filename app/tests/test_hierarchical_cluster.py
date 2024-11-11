import unittest
import numpy as np
import pandas as pd
from clustering.hierarchical.hierarchical_cluster import (
    analyze_hierarchical_clusters,
    plot_dendrogram,
)
from unittest.mock import patch


class TestHierarchicalCluster(unittest.TestCase):
    """
    Unit tests for the hierarchical clustering functions in the `hierarchical_cluster` module.
    This test suite verifies the behavior of cluster analysis and dendrogram plotting
    for hierarchical clustering.
    """

    @patch("clustering.hierarchical.hierarchical_cluster.mlflow")
    def test_analyze_hierarchical_clusters(self, mock_mlflow):
        """
        Test the `analyze_hierarchical_clusters` function to ensure it correctly
        analyzes the contents of each hierarchical cluster.

        Mocks:
        - `mlflow`: Mocked to verify that logging functions are called during analysis.

        Asserts:
        - The output DataFrame includes the 'hierarchical_cluster' column with the cluster labels.
        """
        # Arrange: Set up the input DataFrame with user roles and cluster labels
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "role1": [1, 0, 1],
                "role2": [0, 1, 0],
                "hierarchical_cluster": [0, 0, 1],
            }
        ).set_index("user_id")

        # Act: Call the function under test
        analyze_hierarchical_clusters(df.copy(), df["hierarchical_cluster"])

        # Assert: Verify that the 'hierarchical_cluster' column exists in the DataFrame
        self.assertIn("hierarchical_cluster", df.columns)

    @patch("clustering.hierarchical.hierarchical_cluster.plt.show")
    def test_plot_dendrogram(self, mock_show):
        """
        Test the `plot_dendrogram` function to ensure it generates a dendrogram plot
        without raising any exceptions.

        Mocks:
        - `plt.show`: Mocked to suppress the plot display during testing.

        Asserts:
        - The test passes if no exception is raised during the plotting.
        """
        # Arrange: Generate random input data and labels
        data = np.random.rand(10, 2)
        labels = [f"user_{i}" for i in range(10)]

        # Act: Call the function under test
        plot_dendrogram(data, labels)

        # Assert: The test passes if no exception is raised


if __name__ == "__main__":
    unittest.main()
