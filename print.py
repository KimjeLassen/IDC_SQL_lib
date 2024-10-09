import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import position_roles, user_roles_model, ous_model, positions

pos = positions.get_name_ids_template(10)
ous = ous_model.get_name_and_ids(pos)
for ou in ous:
    for id, name in ou:
        print(f"ID: {id}, Name: {name} with positions:")
        subset_of_A = set([id]) # the subset of A
        results = [a for a in pos if a.ou_id in subset_of_A]
        for res in results:
           urID = position_roles.get_user_role_from_position(res.id)
           if (urID is not None):
                print(f"Position: {res.name}, User Role ID: {urID.user_role_id}")
#
#user_role_id = position_roles.get_user_role_from_position('22a8b2e9-b367-1909-62f8-a3bec3368efb')
#
#user_roles_model.get_id_name_identifier(user_role_id.user_role_id)