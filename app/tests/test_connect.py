from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text  # Import text for raw SQL execution
import pytest
from app.database.db_base import get_engine, get_db


@pytest.fixture
def db_engine():
    """
    Fixture to provide the SQLAlchemy engine.
    """
    return get_engine()


@pytest.fixture
def db_session():
    """
    Fixture to provide a database session.
    """
    session = next(get_db())
    try:
        yield session
    finally:
        session.close()


def test_engine_connection(db_engine):
    """
    Test if the database engine can connect to the database.
    """
    try:
        with db_engine.connect() as connection:
            # Use text() to execute raw SQL in SQLAlchemy 2.x
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1, "Engine connection failed or query did not return the expected result"
    except OperationalError as e:
        pytest.fail(f"Database connection failed: {e}")