from sqlalchemy import Column, Integer, String, SmallInteger, DateTime
from db_base import Base, session

class RoleGroup(Base):
    __tablename__ = 'rolegroup'
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
   # user_only = Column(SmallInteger)
   # ou_inherit_allowed = Column(SmallInteger)
   # last_updated_by = Column(String)
   # last_updated = Column(DateTime)
   # created_by = Column(String)
   # bitmap = Column(Integer)

def print_group_name_and_desc():
    result = session.query(RoleGroup.name, RoleGroup.description).all()
    session.close()
    for name, desc in result:
        if(desc == ""):
            print (f"Rolegroup name: {name}")
        else: 
            print(f"Rolegroup name: {name}, Description: {desc}")

def add_group():
    new_group = RoleGroup(id='newgroup', name='New Group', description='New Group Description', user_only=0, ou_inherit_allowed=0, last_updated_by='admin', last_updated='2021-02-25 00:00:00', created_by='admin', bitmap=0)
    session.add(new_group)
    session.commit()
    session.close()

def delete_group():
    session.query(RoleGroup).filter(RoleGroup.id == 'newgroup').delete()
    session.commit()
    session.close()