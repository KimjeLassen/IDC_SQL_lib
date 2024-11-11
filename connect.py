# connect.py
from sqlalchemy import create_engine
import logging
import pandas as pd
import mlflow
import traceback
from dotenv import load_dotenv
import os

# Load environment variables from .env file for database credentials
load_dotenv()

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
conn_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def engine():
    """
    Initialize and return a SQLAlchemy engine for database connections.

    Returns
    -------
    engine : SQLAlchemy Engine
        A SQLAlchemy engine instance configured with the provided connection string.
    """
    engine = create_engine(conn_string)
    return engine


def fetch_data(sql_query):
    """
    Execute a SQL query and retrieve data from the specified database.

    Parameters
    ----------
    sql_query : str
        The SQL query string to execute.

    Returns
    -------
    DataFrame or None
        A DataFrame containing the retrieved data, or None if an error occurs.

    Logging
    -------
    Logs the shape of the data (number of rows and columns) if data is successfully fetched.
    Logs and tracks errors in MLflow if data retrieval fails.
    """
    try:
        df = pd.read_sql(sql_query, engine())
        logger.info("Data loaded successfully from the database.")
        # Log the shape of the retrieved data for reference
        mlflow.log_param("data_shape", df.shape)
        return df
    except Exception as e:
        # Capture and log any errors that occur during data fetching
        error_trace = traceback.format_exc()
        logger.error("An error occurred while fetching data:", exc_info=True)
        mlflow.log_text(error_trace, "fetch_error_trace.txt")
        mlflow.log_param("fetch_error", str(e))
        return None
