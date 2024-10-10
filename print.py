import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import position_roles, user_roles_model, ous_model, positions
from sqlalchemy.orm import DeclarativeBase

pos = positions.get_name_ids_template(10)
ous = ous_model.get_name_and_ids(pos)

for ou in ous:
    belongTo : list = [positions]
    print(f"OU Name: {ou.name}, OU ID: {ou.id}")
    for position in pos:
        if position.ou_id == ou.id:
            belongTo.append(position)
            pos.remove(position)
    if len(belongTo) == 0:
        print("No positions found")
    else:
        print(f"Position: {len(belongTo)}")

#
#user_role_id = position_roles.get_user_role_from_position('22a8b2e9-b367-1909-62f8-a3bec3368efb')
#
#user_roles_model.get_id_name_identifier(user_role_id.user_role_id)