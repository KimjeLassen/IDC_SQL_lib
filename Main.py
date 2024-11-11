# Main.py
from connect import db_name, fetch_data
from clustering_pipeline import run_pipeline

# Define the SQL query to fetch user roles and system roles from the database
sql_query = f"""
    SELECT 
        urm.user_id,
        sr.name AS system_role_name
    FROM 
        {db_name}.user_roles_mapping urm
    JOIN 
        {db_name}.system_role_assignments sra ON urm.user_role_id = sra.user_role_id
    JOIN 
        {db_name}.system_roles sr ON sra.system_role_id = sr.id;
"""

# Fetch data from the database using the defined SQL query
df = fetch_data(sql_query)

# Run the clustering pipeline with the fetched data
run_pipeline(df)
