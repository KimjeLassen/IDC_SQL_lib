# app/main.py
from fastapi import FastAPI
from app.api import cluster_api

app = FastAPI(
    title="Clustering Microservice",
    description="API for performing clustering on user roles data.",
    version="1.0.0",
)

app.include_router(cluster_api.router, prefix="/api/clustering", tags=["Clustering"])
