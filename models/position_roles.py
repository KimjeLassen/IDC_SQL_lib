from sqlalchemy import Column, Integer, String, SmallInteger, DateTime, JSON, TIMESTAMP, ForeignKey
from db_base import Base, session

class PositionRoles(Base):
    __tablename__ = 'position_roles'
    position_id = Column(String, ForeignKey('positions.id'), primary_key=True)
    user_role_id = Column(String, ForeignKey('userrole.id'), primary_key=True)