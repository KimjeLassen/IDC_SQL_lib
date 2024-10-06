from sqlalchemy import Column, String, ForeignKey, JSON
from db_base import Base, session

class User_Roles_Mapping(Base):
    __tablename__ = 'user_roles_mapping'
    user_id = Column(String, ForeignKey('users.id'))
    user_role_id = Column(String, ForeignKey('userrole.id'))
    meta_data = Column(JSON)