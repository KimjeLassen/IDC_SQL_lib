# app/database/models
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    BigInteger,
    ForeignKey,
    TIMESTAMP,
    JSON,
    DateTime,
    Boolean,
)
from sqlalchemy.orm import relationship
from app.database.db_base import Base
import os

load_dotenv()

DEFAULT_IT_SYSTEM_ID = "00000000-0000-0000-0000-default000000"


class ClusteringRun(Base):
    __tablename__ = "clustering_runs"
    __table_args__ = {"schema": os.getenv("DB_NAME")}

    run_id = Column(String(36), primary_key=True)
    status = Column(String(20), nullable=False)
    algorithm = Column(String(20))
    results = Column(JSON)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))


class UserRoles(Base):
    __tablename__ = "user_roles"
    __table_args__ = {"schema": os.getenv("DB_NAME")}

    id = Column(String(36), primary_key=True)
    name = Column(String(255))
    identifier = Column(String(128))
    description = Column(Text)
    it_system_id = Column(String(36), default=DEFAULT_IT_SYSTEM_ID, nullable=False)
    user_only = Column(Boolean, default=False)
    ou_inherit_allowed = Column(Boolean, default=False)
    delegated_from_cvr = Column(String(8), nullable=True)
    last_updated_by = Column(String(64), nullable=True)
    last_updated = Column(TIMESTAMP)
    created_by = Column(String(64))
    created = Column(TIMESTAMP)
    bitmap = Column(Integer, default=0)

    # Relationship to UserRolesMapping
    user_roles_mappings = relationship(
        "UserRolesMapping", back_populates="role_candidate"
    )


class UserRolesMapping(Base):
    __tablename__ = "user_roles_mapping"
    __table_args__ = {"schema": os.getenv("DB_NAME")}

    user_role_id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)

    # New Foreign Key to UserRoles
    role_candidate_id = Column(
        String(36),
        ForeignKey(f"{os.getenv('DB_NAME')}.user_roles.id"),
        nullable=True,
    )

    # Relationships
    system_role_assignments = relationship(
        "SystemRoleAssignments",
        back_populates="user_role_mapping",
        cascade="all, delete-orphan",
    )

    # Relationship to UserRoles
    role_candidate = relationship("UserRoles", back_populates="user_roles_mappings")


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
    user_role_mapping = relationship(
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
