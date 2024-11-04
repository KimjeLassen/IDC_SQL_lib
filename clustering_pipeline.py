import sys
sys.path.insert(1, 'models')

from db_base import engine
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

def fetch_data(sql_query):
    """
    Fetch data from the database using the provided SQL query.

    Parameters
    ----------
    sql_query : str
        The SQL query string to execute.

    Returns
    -------
    DataFrame or None
        The data retrieved from the database as a pandas DataFrame,
        or None if an error occurs.
    """
    try:
        # Execute the SQL query and load the result into a DataFrame
        df = pd.read_sql(sql_query, engine)
        print("Data loaded successfully from the database.")
        return df
    except Exception as e:
        # Handle exceptions during data fetching
        print("An error occurred while fetching data:", e)
        return None

def transform_to_binary_matrix(df):
    """
    Transform system roles into a binary access matrix.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing 'user_id' and 'system_role_name' columns.

    Returns
    -------
    DataFrame
        A binary access matrix where rows represent users and columns represent system roles.
        Each cell contains 1 if the user has the role, and 0 otherwise.
    """
    # Convert 'system_role_name' to dummy variables (one-hot encoding)
    binary_access_matrix = pd.get_dummies(df, columns=['system_role_name'], prefix='', prefix_sep='')
    # Group by 'user_id' and take the maximum to combine multiple roles per user
    binary_access_matrix = binary_access_matrix.groupby('user_id').max()
    return binary_access_matrix

def analyze_clusters(binary_access_matrix):
    """
    Analyze the contents of each cluster to validate their contents.
    
    Parameters
    ----------
    binary_access_matrix : DataFrame
        The binary access matrix with cluster labels assigned.
    
    Returns
    -------
    None
    """
    # Create 'clusters' directory if it doesn't exist
    os.makedirs('clusters', exist_ok=True)
    
    # Group the data by clusters
    clusters = binary_access_matrix.groupby('cluster')
    
    for cluster_label, cluster_data in clusters:
        print(f"\nCluster {cluster_label}:")
        # Remove 'cluster' column to get only role columns
        cluster_roles = cluster_data.drop('cluster', axis=1)
        
        # Compute the sum of each role in the cluster
        role_sums = cluster_roles.sum()
        # Calculate the percentage of users in the cluster that have each role
        role_percentages = (role_sums / len(cluster_data)) * 100
        
        # Get roles that are common in the cluster (e.g., present in over 50% of users)
        common_roles = role_percentages[role_percentages > 50].sort_values(ascending=False)
        
        print(f"\nNumber of users in cluster: {len(cluster_data)}")
        print("\nCommon roles (present in over 50% of users):")
        print(common_roles)
        
        # List the top N roles
        top_n = 5
        top_roles = role_percentages.sort_values(ascending=False).head(top_n)
        print(f"\nTop {top_n} roles in the cluster:")
        print(top_roles)
        
        # Identify roles unique to this cluster (if any)
        unique_roles = role_percentages[role_percentages == 100]
        if not unique_roles.empty:
            print("\nRoles unique to this cluster (present in all users of the cluster):")
            print(unique_roles)
        
        # Save cluster data to 'clusters' directory
        cluster_data.to_csv(f"clusters/cluster_{cluster_label}_data.csv")

def find_optimal_clusters(data, min_clusters=2, max_clusters=10):
    """
    Determine the optimal number of clusters using the silhouette score.

    Parameters
    ----------
    data : DataFrame or array-like
        The data to cluster.
    min_clusters : int, optional
        The minimum number of clusters to try. Default is 2.
    max_clusters : int, optional
        The maximum number of clusters to try. Default is 10.

    Returns
    -------
    int
        The optimal number of clusters based on the highest silhouette score.
    """
    silhouette_scores = []
    highest_silhouette_score = -1
    optimal_n_clusters = min_clusters
    
    # Iterate over the range of cluster numbers
    for n_clusters in range(min_clusters, max_clusters + 1):
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        # Calculate the average silhouette score for the current number of clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        # Update the optimal number of clusters if the current silhouette score is better
        if silhouette_avg > highest_silhouette_score:
            highest_silhouette_score = silhouette_avg
            optimal_n_clusters = n_clusters
            
    print(f"\nThe optimal number of clusters is: {optimal_n_clusters} with a silhouette score of {highest_silhouette_score}")
    return optimal_n_clusters

def perform_clustering(data, n_clusters):
    """
    Run KMeans and Hierarchical clustering on the data.

    Parameters
    ----------
    data : DataFrame or array-like
        The data to cluster.
    n_clusters : int
        The number of clusters to form.

    Returns
    -------
    tuple of arrays
        kmeans_labels : array
            Cluster labels from KMeans clustering.
        hierarchical_labels : array
            Cluster labels from Agglomerative (Hierarchical) clustering.
    """
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)

    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(data)
    
    return kmeans_labels, hierarchical_labels

def plot_clusters(data, labels, title):
    """
    Plot clusters in 2D space using PCA for dimensionality reduction.

    Parameters
    ----------
    data : DataFrame or array-like
        The data to plot.
    labels : array-like
        Cluster labels for each data point.
    title : str
        Title for the plot.

    Returns
    -------
    None
    """
    # Reduce data to 2 principal components for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # Create a scatter plot of the data points
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar()
    plt.show()

def plot_dendrogram(data, labels):
    """
    Generate a dendrogram for hierarchical clustering.

    Parameters
    ----------
    data : DataFrame or array-like
        The data to cluster and plot.
    labels : list
        Labels for each data point (e.g., user IDs).

    Returns
    -------
    None
    """
    # Perform hierarchical clustering to obtain linkage matrix
    linked = linkage(data, method='average')
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=False)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Users")
    plt.ylabel("Euclidean Distance")
    plt.show()

def run_pipeline(df):
    """
    Execute the full data processing and clustering pipeline.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing 'user_id' and 'system_role_name' columns.

    Returns
    -------
    None
    """
    if df is not None:
        # Transform the data into a binary access matrix
        binary_access_matrix = transform_to_binary_matrix(df)
        # Find the optimal number of clusters
        optimal_cluster_count = find_optimal_clusters(binary_access_matrix)

        # Perform clustering with the optimal number of clusters
        kmeans_labels, hierarchical_labels = perform_clustering(binary_access_matrix, optimal_cluster_count)

        # Assign cluster labels to the DataFrame
        binary_access_matrix['cluster'] = kmeans_labels  # Use kmeans_labels or hierarchical_labels

        # Analyze cluster contents
        analyze_clusters(binary_access_matrix)

        # **Drop 'cluster' column before plotting**
        data_without_cluster = binary_access_matrix.drop('cluster', axis=1)

        # Plot the results of KMeans clustering
        plot_clusters(data_without_cluster, kmeans_labels, "K-Means Clustering")

        # Plot the results of Hierarchical clustering
        plot_clusters(data_without_cluster, hierarchical_labels, "Hierarchical Clustering")

        # Sample a subset of data for dendrogram visualization
        subset_data = data_without_cluster.sample(n=100, random_state=42)  # Adjust sample size as needed

        # Plot the dendrogram for the subset of data
        plot_dendrogram(subset_data, subset_data.index.tolist())

