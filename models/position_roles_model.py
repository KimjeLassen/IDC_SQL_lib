from sqlalchemy import Column, String, ForeignKey
from db_base import Base, session

class PositionRoles(Base):
    __tablename__ = 'position_roles'
    position_id = Column(String, ForeignKey('positions.id'), primary_key=True)
    user_role_id = Column(String, ForeignKey('userrole.id'), primary_key=True)

def get_position_role_from_position(position_id : str): 
    result = session.query(PositionRoles).where(PositionRoles.position_id == position_id).all()
    session.close()
    return result