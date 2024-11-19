from dotenv import load_dotenv
from sqlalchemy import Column, String, Text, Integer, BigInteger, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from app.database.db_base import Base
import os

load_dotenv()


class UserRolesMapping(Base):
    __tablename__ = "user_roles_mapping"
    __table_args__ = {"schema": os.getenv("DB_NAME")}

    user_role_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    user_metadata = Column("metadata", Text)

    # Relationship to SystemRoleAssignments
    system_role_assignments = relationship(
        "SystemRoleAssignments",
        back_populates="user_role",
        cascade="all, delete-orphan",
    )


class SystemRoleAssignments(Base):
    __tablename__ = "system_role_assignments"
    __table_args__ = {"schema": os.getenv("DB_NAME")}

    id = Column(BigInteger, primary_key=True)
    user_role_id = Column(
        String(36),
        ForeignKey(f"{os.getenv('DB_NAME')}.user_roles_mapping.user_role_id"),
        nullable=False,
    )
    system_role_id = Column(
        String(36),
        ForeignKey(f"{os.getenv('DB_NAME')}.system_roles.id"),
        nullable=False,
    )
    created = Column(TIMESTAMP)

    # Relationships
    user_role = relationship(
        "UserRolesMapping", back_populates="system_role_assignments"
    )
    system_role = relationship("SystemRoles", back_populates="assignments")


class SystemRoles(Base):
    __tablename__ = "system_roles"
    __table_args__ = {"schema": os.getenv("DB_NAME")}

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    identifier = Column(String(128))
    description = Column(Text)
    it_system_id = Column(String(36))
    role_type = Column(String(64))
    ad_group_type = Column(Integer)
    dn = Column(String(512))
    bitmap = Column(Integer)
    when_changed = Column(TIMESTAMP)
    when_created = Column(TIMESTAMP)
    created = Column(TIMESTAMP)

    # Relationship to SystemRoleAssignments
    assignments = relationship(
        "SystemRoleAssignments",
        back_populates="system_role",
        cascade="all, delete-orphan",
    )
