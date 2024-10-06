from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import connect

Base = declarative_base()

engine = connect.engine()
Session = sessionmaker(bind=engine)
session = Session()