from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker
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

def engine():
    engine = create_engine(conn_string)
    return engine
