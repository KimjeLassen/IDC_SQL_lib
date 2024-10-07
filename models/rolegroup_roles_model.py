from sqlalchemy import Column, String, ForeignKey
from db_base import Base, session

class RoleGroup_Roles(Base):
    __tablename__ = 'rolegroup_roles'
    rolegroup_id = Column(String, ForeignKey('rolegroup.id'))
    user_role_id = Column(String, ForeignKey('userrole.id'))