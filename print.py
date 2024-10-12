import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import position_roles, user_roles_model, ous_model, positions
from sqlalchemy.orm import DeclarativeBase

pos = positions.get_all_positions(0)
ous = ous_model.get_name_and_ids(pos)
originalList = pos.copy()
all_all_roles = []
positions.map_positions_to_ou(ous, pos)
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