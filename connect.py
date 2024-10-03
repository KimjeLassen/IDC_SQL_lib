from sqlalchemy import create_engine,text
from sqlalchemy import Column, Integer, String, SmallInteger, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
conn_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

engine = create_engine(conn_string)
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class RoleGroup(Base):
    __tablename__ = 'rolegroup'
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    user_only = Column(SmallInteger)
    ou_inherit_allowed = Column(SmallInteger)
    last_updated_by = Column(String)
    last_updated = Column(DateTime)
    created_by = Column(String)
    bitmap = Column(Integer)

result = session.query(RoleGroup.name, RoleGroup.description).all()

for name, desc in result:
    if(desc == ""):
       print (f"Rolegroup name: {name}")
    else: 
        print(f"Rolegroup name: {name}, Description: {desc}")
