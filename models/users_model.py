from sqlalchemy import Integer, Column, String, SmallInteger, JSON, TIMESTAMP
from db_base import Base, session

class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)
    origin_uuid = Column(String)
    identifier = Column(String)
    template_type = Column(String)
    external_id = Column(String)
    national_id = Column(String)
    name = Column(String)
    active = Column(SmallInteger)
    email = Column(String)
    phone = Column(String)
    dn = Column(String)
    meta_data = Column(JSON)
    created = Column(TIMESTAMP)
    last_updated = Column(TIMESTAMP)
    origin_created = Column(TIMESTAMP)
    origin_last_updated = Column(TIMESTAMP)
    origin_bitmap = Column(Integer)
    bitmap = Column(Integer)