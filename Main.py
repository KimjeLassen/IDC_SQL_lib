import sys
sys.path.insert(1, 'models')
from db_base import engine

from models import positions_model, user_roles_model, ous_model
import pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def get_binary_matrix():
    try:
        sql_query = """
        SELECT 
            urm.user_id,
            sr.name AS system_role_name
        FROM 
            idc_data.user_roles_mapping urm
        JOIN 
            idc_data.system_role_assignments sra ON urm.user_role_id = sra.user_role_id
        JOIN 
            idc_data.system_roles sr ON sra.system_role_id = sr.id;
        """

        # Execute SQL query and load data into DataFrame
        df = pd.read_sql(sql_query, engine)
        print("Data loaded successfully from the database.")

        # Step 1: Convert system_role_name to binary columns (one-hot encoding for each role per user)
        binary_access_matrix = pd.get_dummies(df, columns=['system_role_name'], prefix='', prefix_sep='').groupby('user_id').max()
        return binary_access_matrix
    
    except Exception as e:
        print("An error occurred:", e)

def clusters(binary_access_matrix : pd.DataFrame):
    # Step 2: Determine the optimal number of clusters using silhouette score
    range_n_clusters = list(range(2, 11))
    silhouette_scores = []
    optimal_n_clusters = 2
    highest_silhouette_score = -1

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(binary_access_matrix)
        silhouette_avg = silhouette_score(binary_access_matrix, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        if silhouette_avg > highest_silhouette_score:
            highest_silhouette_score = silhouette_avg
            optimal_n_clusters = n_clusters

    # Save the optimal number of clusters as a variable
    optimal_cluster_count = optimal_n_clusters
    print(f"The optimal number of clusters is: {optimal_cluster_count} with a silhouette score of {highest_silhouette_score}")
    return optimal_cluster_count

def get_kmeans_labels(optimal_cluster_count : int, binary_access_matrix : pd.DataFrame):
    # Step 3: Run KMeans clustering with the optimal number of clusters
    optimal_kmeans = KMeans(n_clusters=optimal_cluster_count, random_state=42)
    kmeans_labels = optimal_kmeans.fit_predict(binary_access_matrix)
    return kmeans_labels

def get_hierarchical_labels(optimal_cluster_count : int, binary_access_matrix : pd.DataFrame):
    # Step 4: Run Hierarchical clustering with the optimal number of clusters
    hierarchical = AgglomerativeClustering(n_clusters=optimal_cluster_count)
    hierarchical_labels = hierarchical.fit_predict(binary_access_matrix)
    return hierarchical_labels

def get_reduced_data(binary_access_matrix : pd.DataFrame):
    # Step 5: Reduce dimensions to 2D for visualization using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(binary_access_matrix)
    return reduced_data

    # Plotting function
def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar()
    plt.show()

    # K-Means Clustering Plot

