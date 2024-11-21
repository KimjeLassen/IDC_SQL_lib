# app/database/engine
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os


load_dotenv()

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
conn_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Initialize the engine once at module level
_engine = create_engine(conn_string)


def get_engine():
    """
    Return the initialized SQLAlchemy engine.
    """
    return _engine
