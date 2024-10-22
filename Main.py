import sys
sys.path.insert(1, 'models')
from db_base import engine

from models import positions_model, user_roles_model, ous_model
import pandas as pd

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
def hello_world():
    return"Hello World"

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

    # - Drop duplicate rows to ensure unique position-role combinations
    df_cleaned = df_cleaned.drop_duplicates(subset=['position_id', 'role_name'])

    # Step 3: Remove the 'role_id' column as it's not needed for the encoding
    df_cleaned = df_cleaned.drop(columns=['role_id'])

    # Step 4: Apply multi-hot encoding for the 'role_name' column
    # - This will create binary columns for each role
    multi_hot_encoded_df = pd.get_dummies(df_cleaned, columns=['role_name'], prefix='', prefix_sep='')

    # Step 5: Group by 'position_id' to ensure that each position has one row with aggregated roles
    multi_hot_encoded_df_grouped = multi_hot_encoded_df.groupby('position_id').max().reset_index()

    # Optional Step: Save the cleaned and multi-hot encoded dataset to a CSV file
    multi_hot_encoded_df_grouped.to_csv('multi_hot_encoded_dataset.csv', index=False)