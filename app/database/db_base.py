# app/database/db_base.py
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.database.engine import get_engine


class Base(DeclarativeBase):
    """
    Base class for all ORM models.

    Inherits from `DeclarativeBase`, which provides the foundation for SQLAlchemy's
    ORM functionality. Models created in the application will inherit from `Base`,
    enabling them to map Python classes to database tables.
    """

    pass


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Dependency that provides a SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
