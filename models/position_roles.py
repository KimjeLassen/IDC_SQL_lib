from sqlalchemy import Column, Integer, String, SmallInteger, DateTime, JSON, TIMESTAMP, ForeignKey
from db_base import Base, session

class PositionRoles(Base):
    __tablename__ = 'position_roles'
    position_id = Column(String, ForeignKey('positions.id'), primary_key=True)
    user_role_id = Column(String, ForeignKey('userrole.id'), primary_key=True)

def get_user_role_from_position(position_id : str): 
    result = session.query(PositionRoles.user_role_id).filter(PositionRoles.position_id == position_id).first()
    session.close()
    print(f"User Role ID: {result.user_role_id}")
    return(result)