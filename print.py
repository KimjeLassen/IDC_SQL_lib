# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'models')

from models import rolegroup_model

rolegroup_model.add_group()
rolegroup_model.print_group_name_and_desc()
rolegroup_model.delete_group()