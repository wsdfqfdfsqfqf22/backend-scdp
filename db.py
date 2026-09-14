"""
SCDP — Couche base de données
MySQL / SQLAlchemy — Modèles, CRUD, sessions, transactions
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

# ─────────────────────────────────────────────
# Configuration moteur MySQL
# ─────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://root:@127.0.0.1:3306/scdp?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
    connect_args={
        "connect_timeout": 10,
        "read_timeout": 30,
        "write_timeout": 30,
    },
)

# Force UTF-8 sur chaque connexion
@event.listens_for(engine, "connect")
def set_utf8(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci")
    cursor.execute("SET time_zone = '+00:00'")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ─────────────────────────────────────────────
# Base déclarative
# ─────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# Modèles SQLAlchemy
# ─────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(
        Enum("medecin", "admin", name="user_role"),
        nullable=False,
        default="medecin",
        server_default="medecin",
    )
    is_active = Column(Boolean, nullable=False, default=True, server_default="1")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("AuthSession", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pathology_id = Column(String(100), nullable=False, index=True)
    report_json = Column(JSON, nullable=False)
    pdf_path = Column(String(500), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    user = relationship("User", back_populates="reports")

    __table_args__ = (
        Index("ix_reports_user_pathology", "user_id", "pathology_id"),
        Index("ix_reports_created_at", "created_at"),
    )


class PathologyParams(Base):
    __tablename__ = "pathology_params"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pathology_id = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    sha256 = Column(String(64), nullable=False)
    params_json = Column(JSON, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, server_default="1")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_pathology_params_id_active", "pathology_id", "is_active"),
        Index("ix_pathology_params_sha256", "sha256"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    actor_id = Column(String(36), nullable=True, index=True)
    action = Column(String(200), nullable=False, index=True)
    payload = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    __table_args__ = (
        Index("ix_audit_logs_actor_action", "actor_id", "action"),
        Index("ix_audit_logs_created_at", "created_at"),
    )


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    refresh_token_hash = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, nullable=False, default=False, server_default="0")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index("ix_auth_sessions_token_revoked", "refresh_token_hash", "revoked"),
        Index("ix_auth_sessions_expires", "expires_at"),
    )


# ─────────────────────────────────────────────
# Initialisation schéma
# ─────────────────────────────────────────────

def init_db() -> None:
    """Crée toutes les tables si elles n'existent pas."""
    Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────
# Dependency FastAPI
# ─────────────────────────────────────────────

def get_db():
    """Fournit une session DB via dependency injection FastAPI."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ─────────────────────────────────────────────
# CRUD — Users
# ─────────────────────────────────────────────

def create_user(db: Session, email: str, password_hash: str) -> User:
    user = User(
        email=email.lower().strip(),
        password_hash=password_hash,
        role="medecin",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return (
        db.query(User)
        .filter(User.email == email.lower().strip(), User.is_active == True)
        .first()
    )


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return (
        db.query(User)
        .filter(User.id == user_id, User.is_active == True)
        .first()
    )


def update_user_timestamp(db: Session, user_id: str) -> None:
    db.query(User).filter(User.id == user_id).update(
        {"updated_at": datetime.now(timezone.utc)}
    )
    db.commit()


# ─────────────────────────────────────────────
# CRUD — Reports
# ─────────────────────────────────────────────

def create_report(
    db: Session,
    user_id: str,
    pathology_id: str,
    report_json: dict,
    pdf_path: Optional[str] = None,
) -> Report:
    report = Report(
        user_id=user_id,
        pathology_id=pathology_id,
        report_json=report_json,
        pdf_path=pdf_path,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def update_report_pdf(db: Session, report_id: str, pdf_path: str) -> None:
    db.query(Report).filter(Report.id == report_id).update({"pdf_path": pdf_path})
    db.commit()


def get_reports_by_user(
    db: Session,
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    pathology_filter: Optional[str] = None,
) -> tuple[list[Report], int]:
    query = db.query(Report).filter(Report.user_id == user_id)
    if pathology_filter:
        query = query.filter(Report.pathology_id == pathology_filter)
    total = query.count()
    reports = (
        query.order_by(Report.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return reports, total


def get_all_reports(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    pathology_filter: Optional[str] = None,
    user_filter: Optional[str] = None,
) -> tuple[list[Report], int]:
    query = db.query(Report)
    if pathology_filter:
        query = query.filter(Report.pathology_id == pathology_filter)
    if user_filter:
        query = query.filter(Report.user_id == user_filter)
    total = query.count()
    reports = (
        query.order_by(Report.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return reports, total


def get_report_by_id(db: Session, report_id: str) -> Optional[Report]:
    return db.query(Report).filter(Report.id == report_id).first()


# ─────────────────────────────────────────────
# CRUD — PathologyParams
# ─────────────────────────────────────────────

def get_active_params(db: Session, pathology_id: str) -> Optional[PathologyParams]:
    return (
        db.query(PathologyParams)
        .filter(
            PathologyParams.pathology_id == pathology_id,
            PathologyParams.is_active == True,
        )
        .order_by(PathologyParams.created_at.desc())
        .first()
    )


def upsert_pathology_params(
    db: Session,
    pathology_id: str,
    version: str,
    sha256: str,
    params_json: dict,
) -> PathologyParams:
    # Désactiver l'ancienne version active
    db.query(PathologyParams).filter(
        PathologyParams.pathology_id == pathology_id,
        PathologyParams.is_active == True,
    ).update({"is_active": False})

    pp = PathologyParams(
        pathology_id=pathology_id,
        version=version,
        sha256=sha256,
        params_json=params_json,
        is_active=True,
    )
    db.add(pp)
    db.commit()
    db.refresh(pp)
    return pp


def get_all_active_params(db: Session) -> list[PathologyParams]:
    return (
        db.query(PathologyParams)
        .filter(PathologyParams.is_active == True)
        .all()
    )


# ─────────────────────────────────────────────
# CRUD — AuthSession
# ─────────────────────────────────────────────

def create_auth_session(
    db: Session,
    user_id: str,
    refresh_token_hash: str,
    expires_at: datetime,
) -> AuthSession:
    session = AuthSession(
        user_id=user_id,
        refresh_token_hash=refresh_token_hash,
        expires_at=expires_at,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_valid_session(
    db: Session, refresh_token_hash: str
) -> Optional[AuthSession]:
    now = datetime.now(timezone.utc)
    return (
        db.query(AuthSession)
        .filter(
            AuthSession.refresh_token_hash == refresh_token_hash,
            AuthSession.revoked == False,
            AuthSession.expires_at > now,
        )
        .first()
    )


def revoke_session(db: Session, refresh_token_hash: str) -> None:
    db.query(AuthSession).filter(
        AuthSession.refresh_token_hash == refresh_token_hash
    ).update({"revoked": True})
    db.commit()


def revoke_all_user_sessions(db: Session, user_id: str) -> None:
    db.query(AuthSession).filter(
        AuthSession.user_id == user_id,
        AuthSession.revoked == False,
    ).update({"revoked": True})
    db.commit()


def cleanup_expired_sessions(db: Session) -> int:
    now = datetime.now(timezone.utc)
    result = db.query(AuthSession).filter(
        AuthSession.expires_at <= now
    ).delete(synchronize_session=False)
    db.commit()
    return result


# ─────────────────────────────────────────────
# CRUD — AuditLog
# ─────────────────────────────────────────────

def write_audit_log(
    db: Session,
    action: str,
    actor_id: Optional[str] = None,
    payload: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    log = AuditLog(
        actor_id=actor_id,
        action=action,
        payload=payload,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(log)
    db.commit()


def get_audit_logs(
    db: Session,
    page: int = 1,
    page_size: int = 50,
    actor_filter: Optional[str] = None,
    action_filter: Optional[str] = None,
) -> tuple[list[AuditLog], int]:
    query = db.query(AuditLog)
    if actor_filter:
        query = query.filter(AuditLog.actor_id == actor_filter)
    if action_filter:
        query = query.filter(AuditLog.action.like(f"%{action_filter}%"))
    total = query.count()
    logs = (
        query.order_by(AuditLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return logs, total