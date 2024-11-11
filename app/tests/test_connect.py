import unittest
from unittest.mock import patch, MagicMock
from database.connect import engine, fetch_data


class TestConnect(unittest.TestCase):
    """
    Unit tests for the database connection module (`connect.py`).
    This test suite verifies the behavior of the `engine` and `fetch_data` functions,
    ensuring they handle database connections and data retrieval correctly.
    """

    @patch("database.connect.create_engine")
    def test_engine(self, mock_create_engine):
        """
        Test the `engine` function to ensure it correctly initializes and returns
        a SQLAlchemy engine instance.

        Mocks:
        - `create_engine`: Mocked to simulate the creation of a database engine.

        Asserts:
        - Checks that `create_engine` is called once.
        - Verifies that the returned engine instance matches the mocked instance.
        """
        # Arrange: Set up the mock for `create_engine`
        mock_engine_instance = MagicMock()
        mock_create_engine.return_value = mock_engine_instance

        # Act: Call the function under test
        eng = engine()

        # Assert: Verify the expected behavior
        mock_create_engine.assert_called_once()
        self.assertEqual(eng, mock_engine_instance)

    @patch("database.connect.pd.read_sql")
    @patch("database.connect.engine")
    def test_fetch_data_success(self, mock_engine, mock_read_sql):
        """
        Test the `fetch_data` function for successful data retrieval.

        Mocks:
        - `pd.read_sql`: Mocked to simulate reading data from the database.
        - `engine`: Mocked to provide a database connection.

        Asserts:
        - Checks that `pd.read_sql` is called once with the correct SQL query and engine.
        - Verifies that the DataFrame returned by `fetch_data` matches the mocked DataFrame.
        """
        # Arrange: Set up the mock DataFrame and return value
        mock_df = MagicMock()
        mock_read_sql.return_value = mock_df
        sql_query = "SELECT * FROM users"

        # Act: Call the function under test
        df = fetch_data(sql_query)

        # Assert: Verify the expected behavior
        mock_read_sql.assert_called_once_with(sql_query, mock_engine())
        self.assertEqual(df, mock_df)

    @patch("database.connect.mlflow")
    @patch("database.connect.logger")
    @patch("database.connect.pd.read_sql")
    @patch("database.connect.engine")
    def test_fetch_data_exception(
        self, mock_engine, mock_read_sql, mock_logger, mock_mlflow
    ):
        """
        Test the `fetch_data` function when an exception occurs during data retrieval.

        Mocks:
        - `pd.read_sql`: Mocked to raise an exception to simulate a database error.
        - `logger`: Mocked to verify error logging.
        - `mlflow`: Mocked to verify logging of error details in MLflow.

        Asserts:
        - Checks that `fetch_data` returns `None` when an exception is raised.
        - Verifies that the logger's `error` method is called to log the error.
        - Confirms that `mlflow.log_text` and `mlflow.log_param` are called with the correct arguments.
        """
        # Arrange: Set up the mock to raise an exception
        mock_read_sql.side_effect = Exception("Database Error")
        sql_query = "SELECT * FROM users"

        # Act: Call the function under test
        df = fetch_data(sql_query)

        # Assert: Verify the expected behavior when an exception occurs
        self.assertIsNone(df)
        mock_logger.error.assert_called()
        mock_mlflow.log_text.assert_called()
        mock_mlflow.log_param.assert_called_with("fetch_error", "Database Error")


if __name__ == "__main__":
    unittest.main()
