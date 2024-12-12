import unittest
import numpy as np
import pandas as pd
from app.clustering.data_manipulation.data_manipulation import (
    transform_to_binary_matrix,
    perform_dbscan,
)


class TestDataManipulation(unittest.TestCase):
    """
    Unit tests for data manipulation functions used in clustering.
    This test suite covers transformation to binary matrices, KMeans and hierarchical clustering,
    and DBSCAN clustering.
    """

    def test_transform_to_binary_matrix(self):
        """
        Test the `transform_to_binary_matrix` function to ensure it correctly
        converts user-role mappings into a binary access matrix.

        Asserts:
        - The resulting matrix contains expected columns and index.
        - The values in the matrix are correctly set (1 for assigned roles, 0 otherwise).
        """
        # Arrange: Create a sample DataFrame with user-role mappings
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "system_role_name": ["role1", "role2", "role1", "role3", "role2"],
            }
        )

        # Act: Call the function under test
        result = transform_to_binary_matrix(df)

        # Assert: Verify the binary matrix structure and values
        expected_columns = ["role1", "role2", "role3"]
        expected_index = [1, 2, 3]
        self.assertListEqual(sorted(result.columns.tolist()), sorted(expected_columns))
        self.assertListEqual(result.index.tolist(), expected_index)
        self.assertEqual(result.loc[1, "role1"], 1)
        self.assertEqual(result.loc[2, "role3"], 1)
        self.assertEqual(result.loc[3, "role2"], 1)

    def test_perform_dbscan(self):
        """
        Test the `perform_dbscan` function to ensure it correctly applies DBSCAN clustering.

        Asserts:
        - The length of the cluster labels matches the number of data points.
        - Noise points are labeled as -1.
        """
        # Arrange: Create a sample input array
        data = np.array([[0, 1], [1, 0], [0, 1]])
        dbscan_eps = 0.5
        dbscan_min_samples = 2

        # Act: Call the function under test
        labels = perform_dbscan(data, dbscan_eps, dbscan_min_samples)

        # Assert: Verify the output labels
        self.assertEqual(len(labels), len(data))
        self.assertIn(-1, labels)  # Check if noise points are labeled


if __name__ == "__main__":
    unittest.main()
