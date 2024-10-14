from sqlalchemy import Column, Integer, String, SmallInteger, TIMESTAMP
from db_base import Base, session

class UserRole(Base):
    __tablename__ = 'user_roles'
    id = Column(String, primary_key=True)
    name = Column(String)
    identifier = Column(String)
    description = Column(String)
    it_system_id = Column(String)
    user_only = Column(SmallInteger)
    ou_inherit_allowed = Column(SmallInteger)
    delegated_from_cvr = Column(String)
    last_updated_by = Column(String)
    last_updated = Column(TIMESTAMP)
    created_by = Column(String)
    created = Column(TIMESTAMP)
    bitmap = Column(Integer)

def get_group_from_id(id : str):
    result = session.query(UserRole).where(UserRole.id == id).first()
    session.close()
    return result