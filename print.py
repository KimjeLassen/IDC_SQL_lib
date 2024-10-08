# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'models')

from models import position_roles, user_roles_model, ous_model, positions

pos = positions.get_name_ids_template()

ous_model.get_name_and_ids(pos)
#
#user_role_id = position_roles.get_user_role_from_position('22a8b2e9-b367-1909-62f8-a3bec3368efb')
#
#user_roles_model.get_id_name_identifier(user_role_id.user_role_id)