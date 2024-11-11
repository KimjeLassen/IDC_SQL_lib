#db_base.py
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import connect

class Base(DeclarativeBase):
    """
    Base class for all ORM models.
    
    Inherits from `DeclarativeBase`, which provides the foundation for SQLAlchemy's 
    ORM functionality. Models created in the application will inherit from `Base`, 
    enabling them to map Python classes to database tables.
    """
    pass

# Initialize the database engine using the connection details specified in `connect.py`
engine = connect.engine()

# Create a session factory bound to the engine
Session = sessionmaker(bind=engine)

# Instantiate a session for executing database operations
session = Session()
