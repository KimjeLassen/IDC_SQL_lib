from sqlalchemy.orm import sessionmaker, DeclarativeBase
import connect

class Base(DeclarativeBase):
    pass

engine = connect.engine()
Session = sessionmaker(bind=engine)
session = Session()