from sqlalchemy import create_engine,text
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

with engine.connect() as connection:
    result = connection.execute(text("SELECT * FROM rolegroup"))
    print(result.first())
