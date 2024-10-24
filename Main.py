import sys
sys.path.insert(1, 'models')
from db_base import engine

from models import positions_model, user_roles_model, ous_model
import pandas as pd
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt

pos = positions_model.get_all_positions(10)
ous = ous_model.get_name_and_ids(pos)
originalList = pos.copy()
all_all_roles = []
#for positions in pos:
#    print(f"Position name: {positions.name}, ou_id: {positions.ou_id}")
positions_model.map_positions_to_ou(ous, pos)
#for ou in ous:
#    belongTo : list = []
#    print(f"OU Name: {ou.name}, OU ID: {ou.id}")
#    for position in pos:
#        if position.ou_id == ou.id:
#            belongTo.append(position)
#            pos.remove(position)
#    if len(belongTo) == 0:
#        print("No positions found")
#    else:
#        print(f"Position: {len(belongTo)}")
#        for po in belongTo:
#            user_role_ids = position_roles.get_user_role_from_position(po.id)
#            if (len(user_role_ids) == 0):
#                break
#            else:
#                all_all_roles.append(user_role_ids)
#                for role_id in user_role_ids:
#                       print(role_id.user_role_id)
#                       user_role = user_roles_model.get_group_from_id(role_id.user_role_id) 
#                       print(f"Role: {user_role.name}")
#    print("")
#print(f"Amount of positions: {len(originalList)}")
#print(f"Amount of roles: {len(all_all_roles)}")
#print(f"Amount of organizations {len(ous)}")
#
#user_role_id = position_roles.get_user_role_from_position('22a8b2e9-b367-1909-62f8-a3bec3368efb')
#
#user_roles_model.get_id_name_identifier(user_role_id.user_role_id)

def clean_save_csv():
    sql_query = """
    SELECT DISTINCT
        p.id AS position_id,
        p.name AS position_name,
        p.ou_id,
        o.name AS ou_name,
        ur.id AS role_id,
        ur.name AS role_name
    FROM korsbaek.positions p
    JOIN korsbaek.ous o ON p.ou_id = o.id
    JOIN korsbaek.position_roles pr ON p.id = pr.position_id
    JOIN korsbaek.user_roles ur ON pr.user_role_id = ur.id;
    """
    
    df = pd.read_sql(sql_query, engine)
    df_cleaned = df.dropna(subset=['position_name', 'role_name'])
    df_cleaned = df_cleaned.drop_duplicates(subset=['position_id', 'role_name'])
    df_cleaned = df_cleaned.drop(columns=['role_id'])

    # Apply multi-hot encoding for the 'role_name' column
    multi_hot_encoded_df = pd.get_dummies(df_cleaned, columns=['role_name'], prefix='', prefix_sep='')

    # Group by 'position_id' to ensure each position has one row with aggregated roles
    multi_hot_encoded_df_grouped = multi_hot_encoded_df.groupby('position_id').max().reset_index()

    # Save the cleaned and multi-hot encoded dataset to a CSV file
    csv_file_path = 'multi_hot_encoded_dataset.csv'
    multi_hot_encoded_df_grouped.to_csv(csv_file_path, index=False)
    
    return csv_file_path

def optimal_clusters(file_path, max_clusters=20, alpha=0.5, plot=True):
    """
    Analyze the optimal number of clusters in a dataset using both the Elbow and Silhouette methods.
    
    :param file_path: Path to the CSV file containing the dataset.
    :param max_clusters: Maximum number of clusters to consider (default is 20).
    :param alpha: Weight to give the Silhouette method when combining the results (default is 0.5).
    :param plot: Whether to plot the results (default is True).
    :return: Tuple containing the optimal number of clusters from the Elbow method, Silhouette method, and the combined result.
    """
    
    # Load the dataset
    def load_dataset(file_path):
        data = pd.read_csv(file_path)
        # Assuming binary columns are the ones to use for clustering
        binary_columns = data.select_dtypes(include=['bool'])
        X = binary_columns.astype(int)  # Convert True/False to 1/0
        return X

    # Elbow method using kneed to find the optimal number of clusters
    def find_best_k_elbow(X, max_clusters=20):
        options = range(2, max_clusters)
        inertias = []

        for n_clusters in options:
            model = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            inertias.append(model.inertia_)

        knee_locator = KneeLocator(options, inertias, curve="convex", direction="decreasing")
        best_k_elbow = knee_locator.elbow

        print(f"Elbow Method - Optimal number of clusters: {best_k_elbow}")

        # Plot inertia vs number of clusters
        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(options, inertias, '-o')
            plt.axvline(best_k_elbow, color='red', linestyle='--')
            plt.title("Elbow Method (Inertia)")
            plt.xlabel("No. of clusters (K)")
            plt.ylabel("Inertia")
            plt.show()

        return best_k_elbow

    # Silhouette method to find the optimal number of clusters
    def find_best_k_silhouette(X, max_clusters=20):
        options = range(2, max_clusters)
        silhouette_scores = []

        for n_clusters in options:
            model = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            labels = model.labels_
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

        # Plot silhouette scores
        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(options, silhouette_scores, '-o')
            plt.axvline(np.argmax(silhouette_scores) + 2, color='red', linestyle='--')  # Offset +2 because range starts at 2
            plt.title("Silhouette Method")
            plt.xlabel("No. of clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.show()

        best_k_silhouette = options[np.argmax(silhouette_scores)]
        print(f"Silhouette Method - Optimal number of clusters: {best_k_silhouette}")
        return best_k_silhouette

    # Combine both methods by weighting the silhouette result against the elbow result
    def combine_elbow_and_silhouette(K_elbow, K_silhouette, alpha=0.5):
        K_final = K_elbow + alpha * (K_silhouette - K_elbow)
        return round(K_final)

    # Main logic
    X = load_dataset(file_path)
    best_k_elbow = find_best_k_elbow(X, max_clusters)
    best_k_silhouette = find_best_k_silhouette(X, max_clusters)
    best_k_combined = combine_elbow_and_silhouette(best_k_elbow, best_k_silhouette, alpha)

    if plot:
        # Plot inertia vs number of clusters for elbow method
        plt.figure(figsize=(10, 5))
        
        # Elbow Method plot (inertia)
        plt.subplot(1, 2, 1)
        inertias = []
        options = range(2, max_clusters)
        for n_clusters in options:
            model = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            inertias.append(model.inertia_)
        plt.plot(options, inertias, '-o')
        plt.axvline(best_k_elbow, color='red', linestyle='--')
        plt.title("Elbow Method (Inertia)")
        plt.xlabel("No. of clusters (K)")
        plt.ylabel("Inertia")
        
        # Silhouette Method plot (silhouette scores)
        plt.subplot(1, 2, 2)
        silhouette_scores = []
        for n_clusters in options:
            model = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            labels = model.labels_
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        plt.plot(options, silhouette_scores, '-o')
        plt.axvline(best_k_silhouette, color='red', linestyle='--')
        plt.title("Silhouette Method")
        plt.xlabel("No. of clusters (K)")
        plt.ylabel("Silhouette Score")
        
        plt.tight_layout()
        plt.show()

    print(f"Combined Method - Final number of clusters: {best_k_combined}")
    return best_k_elbow, best_k_silhouette, best_k_combined
