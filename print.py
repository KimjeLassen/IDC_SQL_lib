import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import rolegroup_model

rolegroup_model.add_group()
rolegroup_model.print_group_name_and_desc()
rolegroup_model.delete_group()