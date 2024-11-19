# app/main.py
from dotenv import load_dotenv
from fastapi import FastAPI
from app.api import clustering


load_dotenv()

app = FastAPI(
    title="Clustering Microservice",
    description="API for performing clustering on user roles data.",
    version="1.0.0",
)

app.include_router(clustering.router, prefix="/api/clustering", tags=["Clustering"])
