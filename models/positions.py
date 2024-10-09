from sqlalchemy import Column, Integer, String, SmallInteger, DateTime, JSON, TIMESTAMP, ForeignKey
from db_base import Base, session

class Positions(Base):
    __tablename__ = 'positions'
    id = Column(String, primary_key=True)
    external_id = Column(String)
    user_id = Column(String, ForeignKey('users.id'))
    external_user_id = Column(String)
    ou_id = Column(String, ForeignKey('ous.id'))
    external_ou_id = Column(String)
    name = Column(String)
    manager = Column(SmallInteger)
    template_type = Column(String)
    hire_date = Column(DateTime)
    leave_date = Column(DateTime)
    meta_data = Column(JSON)
    created = Column(TIMESTAMP)
    last_updated = Column(TIMESTAMP)
    origin_created = Column(TIMESTAMP)
    origin_last_updated = Column(TIMESTAMP)
    bitmap = Column(Integer)

def get_name_ids_template(amount : float):
    result : list = [Positions]
    result = session.query(Positions).limit(amount).all()
    session.close()
    #for name, ou_id in result:
#        print(f"Position name: {name}, ou_id: {ou_id}")
    return result

