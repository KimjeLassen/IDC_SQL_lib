import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import position_roles, user_roles_model, ous_model, positions
from sqlalchemy.orm import DeclarativeBase

pos = positions.get_name_ids_template(6641)
ous = ous_model.get_name_and_ids(pos)
originalList = pos.copy()
all_all_roles = []
for ou in ous:
    belongTo : list = []
    print(f"OU Name: {ou.name}, OU ID: {ou.id}")
    for position in pos:
        if position.ou_id == ou.id:
            belongTo.append(position)
            pos.remove(position)
    if len(belongTo) == 0:
        print("No positions found")
    else:
        print(f"Position: {len(belongTo)}")
        all_roles = []
        for po in belongTo:
            user_role_ids = position_roles.get_user_role_from_position(po.id)
            if (len(user_role_ids) == 0):
                break
            else:
                all_roles.append(user_role_ids)
                for role_id in user_role_ids:
                    print(f"User role id: {role_id}")
        if len(all_roles) == 0:
            print("No roles found")
        else:
            all_all_roles.append(all_roles)            
    print("")
print(len(originalList))
print(len(all_all_roles))
print(len(ous))
#
#user_role_id = position_roles.get_user_role_from_position('22a8b2e9-b367-1909-62f8-a3bec3368efb')
#
#user_roles_model.get_id_name_identifier(user_role_id.user_role_id)