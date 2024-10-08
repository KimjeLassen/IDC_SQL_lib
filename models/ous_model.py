from sqlalchemy import Column, Integer, String, SmallInteger, TIMESTAMP, BigInteger, JSON, ForeignKey
from db_base import Base, session
from collections.abc import Sequence

class Ous(Base):
    __tablename__ = 'ous'
    id = Column(String, primary_key=True)
    external_id = Column(String)
    organisation_id = Column(BigInteger, ForeignKey('organisations.id'))
    name = Column(String)
    short_name = Column(String)
    parent_id = Column(String)
    manager_position_id = Column(String, ForeignKey('positions.id'))
    active = Column(SmallInteger)
    dn = Column(String)
    meta_data = Column(JSON)
    created = Column(TIMESTAMP)
    last_updated = Column(TIMESTAMP)
    origin_created = Column(TIMESTAMP)
    origin_last_updated = Column(TIMESTAMP)
    bitmap = Column(Integer)


def get_name_and_ids(pos: Sequence[positions]):
    for position in pos:
        print(f"Position name: {position.name}, Identifier: {position.id}")
    result = session.query(Ous.id, Ous.name).all()
    session.close()
    #for id, name in result:
        #print(f"Rolegroup name: {name}, Identifier: {id}")