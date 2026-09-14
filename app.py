"""
SCDP — Application FastAPI
Endpoints REST, middlewares, auth JWT, sécurité, startup/shutdown
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # Charge .env depuis le répertoire courant — doit être avant tout os.environ

import hashlib
import os
import sys
import time
import types
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


# ─────────────────────────────────────────────
import bcrypt as _bcrypt_module  # noqa: E402

_bcrypt_about = sys.modules.get("bcrypt.__about__")
if _bcrypt_about is None:
    _bcrypt_about = types.ModuleType("bcrypt.__about__")
    _bcrypt_about.__version__ = getattr(_bcrypt_module, "__version__", "4.0.1")  # type: ignore[attr-defined]
    sys.modules["bcrypt.__about__"] = _bcrypt_about
    _bcrypt_module.__about__ = _bcrypt_about  # type: ignore[attr-defined]

del _bcrypt_module, _bcrypt_about  

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from sqlalchemy.orm import Session

import core
import db as database

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SECRET_KEY = os.environ.get("SCDP_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "Variable d'environnement SCDP_SECRET_KEY manquante. "
        "Définissez-la dans votre fichier .env avant de démarrer l'application."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
APP_START_TIME = time.time()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
logger = structlog.get_logger("scdp.app")



def _prehash_password(password: str) -> str:
    """Pré-hash SHA-256 hex du mot de passe (→ 64 bytes ASCII, jamais > 72)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    """Hash sécurisé : SHA-256 → bcrypt. À utiliser dans /register."""
    return pwd_context.hash(_prehash_password(password))


def verify_password(plain_password: str, hashed: str) -> bool:
    """Vérification : pré-hash SHA-256 → bcrypt.verify. À utiliser dans /login."""
    return pwd_context.verify(_prehash_password(plain_password), hashed)

# ─────────────────────────────────────────────
# Rate limiting en mémoire
# ─────────────────────────────────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(key: str, max_requests: int, window_seconds: int) -> bool:
    """Retourne True si la limite est dépassée."""
    now = time.time()
    timestamps = _rate_limit_store[key]
    # Nettoie les anciennes entrées
    _rate_limit_store[key] = [t for t in timestamps if now - t < window_seconds]
    if len(_rate_limit_store[key]) >= max_requests:
        return True
    _rate_limit_store[key].append(now)
    return False


# ─────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="SCDP API",
    description="Système de Classification Diagnostique Probabiliste — Brevet FR2603250",
    version=APP_VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

_cors_raw = os.environ.get(
    "SCDP_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
)
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Middlewares
# ─────────────────────────────────────────────

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms}ms"
    logger.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
        request_id=request_id,
    )
    return response


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    database.init_db()
    _seed_pathology_params()
    logger.info("scdp.startup", version=APP_VERSION)


def _seed_pathology_params():
    """Insère ou met à jour les paramètres de pathologies en base si nécessaire."""
    db = database.SessionLocal()
    try:
        for pid, config in core.PATHOLOGY_CONFIGS.items():
            sha256 = core.compute_sha256(config)
            existing = database.get_active_params(db, pid)
            if existing is None or existing.sha256 != sha256:
                database.upsert_pathology_params(
                    db=db,
                    pathology_id=pid,
                    version=config["params_version"],
                    sha256=sha256,
                    params_json=config,
                )
                logger.info("scdp.params.seeded", pathology_id=pid, version=config["params_version"])
    finally:
        db.close()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("scdp.shutdown")


# ─────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────

def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ─────────────────────────────────────────────
# Auth dependencies
# ─────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(database.get_db),
) -> database.User:
    payload = decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Type de token invalide.")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token sans sujet.")
    user = database.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable.")
    return user


def require_admin(current_user: database.User = Depends(get_current_user)) -> database.User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs.")
    return current_user


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Le mot de passe doit contenir au moins une majuscule.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Le mot de passe doit contenir au moins un chiffre.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    # Limite anti-DoS : cohérent avec le pré-hash SHA-256 qui absorbe toute taille.
    password: str = Field(..., min_length=1, max_length=1024)


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    created_at: datetime


class EvaluateRequest(BaseModel):
    pathology_id: str = Field(..., min_length=1, max_length=100)
    clinical_data: dict[str, Any] = Field(...)

    @field_validator("pathology_id")
    @classmethod
    def validate_pathology_id(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in core.list_pathology_ids():
            raise ValueError(
                f"Pathologie inconnue : '{v}'. Disponibles : {core.list_pathology_ids()}"
            )
        return v


class BatchEvaluateRequest(BaseModel):
    items: list[EvaluateRequest] = Field(..., min_length=1, max_length=100)


class ReportSummary(BaseModel):
    id: str
    pathology_id: str
    decision_class: str
    score_final_normalized: float
    probability_pct: float
    created_at: datetime
    pdf_available: bool


class PaginatedReports(BaseModel):
    items: list[ReportSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class AuditLogResponse(BaseModel):
    id: int
    actor_id: Optional[str]
    action: str
    payload: Optional[dict]
    ip_address: Optional[str]
    created_at: datetime


class PaginatedAuditLogs(BaseModel):
    items: list[AuditLogResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# ─────────────────────────────────────────────
# Helper: construire ReportSummary
# ─────────────────────────────────────────────

def _report_to_summary(r: database.Report) -> ReportSummary:
    rj = r.report_json or {}
    return ReportSummary(
        id=r.id,
        pathology_id=r.pathology_id,
        decision_class=rj.get("decision_class", "INCERTAIN"),
        score_final_normalized=rj.get("score_final_normalized", 0.0),
        probability_pct=rj.get("probability_pct", 0.0),
        created_at=r.created_at,
        pdf_available=r.pdf_path is not None,
    )


# ─────────────────────────────────────────────
# ENDPOINTS AUTH
# ─────────────────────────────────────────────

@app.post(
    "/api/v1/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Auth"],
    summary="Inscription — crée un compte médecin",
)
async def register(
    body: RegisterRequest,
    request: Request,
    db: Session = Depends(database.get_db),
):
    ip = get_client_ip(request)

    # Rate limiting anti brute-force inscription
    if check_rate_limit(f"register:{ip}", max_requests=10, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Trop de tentatives. Réessayez dans 1 heure.")

    existing = database.get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Cette adresse email est déjà utilisée.")

    hashed = hash_password(body.password)
    user = database.create_user(db=db, email=body.email, password_hash=hashed)

    database.write_audit_log(
        db=db,
        action="auth.register",
        actor_id=user.id,
        payload={"email": user.email},
        ip_address=ip,
        user_agent=request.headers.get("User-Agent"),
    )

    core.ACTIVE_USERS.inc()
    logger.info("auth.register", user_id=user.id, email=user.email)

    return UserResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        created_at=user.created_at,
    )


@app.post(
    "/api/v1/auth/login",
    response_model=TokenResponse,
    tags=["Auth"],
    summary="Connexion — retourne access + refresh token",
)
async def login(
    body: LoginRequest,
    request: Request,
    db: Session = Depends(database.get_db),
):
    ip = get_client_ip(request)

    # Rate limiting anti brute-force
    if check_rate_limit(f"login:{ip}", max_requests=20, window_seconds=900):
        raise HTTPException(status_code=429, detail="Trop de tentatives. Réessayez dans 15 minutes.")
    if check_rate_limit(f"login_email:{body.email}", max_requests=10, window_seconds=900):
        raise HTTPException(status_code=429, detail="Compte temporairement verrouillé. Réessayez dans 15 minutes.")

    user = database.get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.password_hash):
        database.write_audit_log(
            db=db,
            action="auth.login.failed",
            payload={"email": body.email},
            ip_address=ip,
        )
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")

    access_token = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id)
    refresh_hash = core.compute_refresh_token_hash(refresh_token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    database.create_auth_session(
        db=db,
        user_id=user.id,
        refresh_token_hash=refresh_hash,
        expires_at=expires_at,
    )

    database.write_audit_log(
        db=db,
        action="auth.login.success",
        actor_id=user.id,
        payload={"email": user.email},
        ip_address=ip,
        user_agent=request.headers.get("User-Agent"),
    )

    logger.info("auth.login", user_id=user.id)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@app.post(
    "/api/v1/auth/refresh",
    response_model=TokenResponse,
    tags=["Auth"],
    summary="Rafraîchissement du token d'accès",
)
async def refresh_token(
    body: RefreshRequest,
    request: Request,
    db: Session = Depends(database.get_db),
):
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Token de rafraîchissement invalide.")

    token_hash = core.compute_refresh_token_hash(body.refresh_token)
    session = database.get_valid_session(db, token_hash)
    if not session:
        raise HTTPException(status_code=401, detail="Session expirée ou révoquée.")

    user = database.get_user_by_id(db, session.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable.")

    # Rotation du refresh token
    database.revoke_session(db, token_hash)
    new_access = create_access_token(user.id, user.role)
    new_refresh = create_refresh_token(user.id)
    new_refresh_hash = core.compute_refresh_token_hash(new_refresh)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    database.create_auth_session(db, user.id, new_refresh_hash, expires_at)

    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@app.get(
    "/api/v1/auth/me",
    response_model=UserResponse,
    tags=["Auth"],
    summary="Profil utilisateur courant",
)
async def get_me(current_user: database.User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role,
        created_at=current_user.created_at,
    )


# ─────────────────────────────────────────────
# ENDPOINTS SCDP
# ─────────────────────────────────────────────

@app.post(
    "/api/v1/scdp/evaluate",
    tags=["SCDP"],
    summary="Évaluation SCDP — pipeline complet étapes a→g",
)
async def evaluate(
    body: EvaluateRequest,
    request: Request,
    current_user: database.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    ip = get_client_ip(request)

    # Rate limiting
    if check_rate_limit(f"evaluate:{current_user.id}", max_requests=100, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Quota d'évaluations horaire dépassé.")

    config = core.get_pathology_config(body.pathology_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Pathologie '{body.pathology_id}' inconnue.")

    # Validation des données cliniques
    errors = core.validate_clinical_data(body.pathology_id, body.clinical_data, config)
    if errors:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})

    # Pipeline SCDP
    start = time.time()
    loaded_at = datetime.now(timezone.utc)
    engine = core.get_engine_for_pathology(body.pathology_id, loaded_at)

    try:
        report = engine.run_pipeline(body.clinical_data)
    except Exception as exc:
        core.SCDP_ERRORS_TOTAL.labels(error_type="pipeline_error").inc()
        logger.error("scdp.pipeline.error", error=str(exc), pathology_id=body.pathology_id)
        raise HTTPException(status_code=500, detail="Erreur interne du moteur SCDP.") from exc

    duration = time.time() - start
    core.SCDP_EVALUATION_DURATION.labels(pathology_id=body.pathology_id).observe(duration)
    core.SCDP_EVALUATIONS_TOTAL.labels(
        pathology_id=body.pathology_id,
        decision_class=report["decision_class"],
    ).inc()

    # Sauvegarde en base
    db_report = database.create_report(
        db=db,
        user_id=current_user.id,
        pathology_id=body.pathology_id,
        report_json=report,
    )

    # Génération PDF (asynchrone best-effort)
    pdf_path = None
    try:
        pdf_start = time.time()
        pdf_path = core.generate_pdf_report(report, db_report.id)
        database.update_report_pdf(db, db_report.id, pdf_path)
        core.PDF_GENERATION_DURATION.observe(time.time() - pdf_start)
    except Exception as exc:
        logger.warning("scdp.pdf.generation.failed", report_id=db_report.id, error=str(exc))

    database.write_audit_log(
        db=db,
        action="scdp.evaluate",
        actor_id=current_user.id,
        payload={
            "report_id": db_report.id,
            "pathology_id": body.pathology_id,
            "decision_class": report["decision_class"],
        },
        ip_address=ip,
    )

    return {
        "report_id": db_report.id,
        "pdf_available": pdf_path is not None,
        **report,
    }


@app.post(
    "/api/v1/scdp/batch",
    tags=["SCDP"],
    summary="Évaluation en lot (max 100 items)",
)
async def batch_evaluate(
    body: BatchEvaluateRequest,
    request: Request,
    current_user: database.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    ip = get_client_ip(request)
    if check_rate_limit(f"batch:{current_user.id}", max_requests=10, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Quota batch horaire dépassé.")

    core.SCDP_BATCH_SIZE.observe(len(body.items))
    results = []
    loaded_at = datetime.now(timezone.utc)

    for idx, item in enumerate(body.items):
        config = core.get_pathology_config(item.pathology_id)
        if config is None:
            results.append({
                "index": idx,
                "success": False,
                "error": f"Pathologie '{item.pathology_id}' inconnue.",
            })
            continue

        errors = core.validate_clinical_data(item.pathology_id, item.clinical_data, config)
        if errors:
            results.append({
                "index": idx,
                "success": False,
                "pathology_id": item.pathology_id,
                "validation_errors": errors,
            })
            continue

        try:
            engine = core.get_engine_for_pathology(item.pathology_id, loaded_at)
            report = engine.run_pipeline(item.clinical_data)
            db_report = database.create_report(
                db=db,
                user_id=current_user.id,
                pathology_id=item.pathology_id,
                report_json=report,
            )
            results.append({
                "index": idx,
                "success": True,
                "report_id": db_report.id,
                **report,
            })
            core.SCDP_EVALUATIONS_TOTAL.labels(
                pathology_id=item.pathology_id,
                decision_class=report["decision_class"],
            ).inc()
        except Exception as exc:
            core.SCDP_ERRORS_TOTAL.labels(error_type="batch_item_error").inc()
            results.append({
                "index": idx,
                "success": False,
                "pathology_id": item.pathology_id,
                "error": "Erreur interne du moteur SCDP.",
            })
            logger.error("scdp.batch.item.error", index=idx, error=str(exc))

    database.write_audit_log(
        db=db,
        action="scdp.batch",
        actor_id=current_user.id,
        payload={"count": len(body.items)},
        ip_address=ip,
    )

    return {"count": len(body.items), "results": results}


@app.get(
    "/api/v1/scdp/pathologies",
    tags=["SCDP"],
    summary="Liste des pathologies disponibles",
)
async def list_pathologies(current_user: database.User = Depends(get_current_user)):
    result = []
    for pid, config in core.PATHOLOGY_CONFIGS.items():
        result.append({
            "pathology_id": pid,
            "pathology_label": config["pathology_label"],
            "params_version": config["params_version"],
            "reference_standard": config.get("reference_standard", ""),
            "ic_required": config["ic_required"],
            "specific_fields": [
                {"name": f["name"], "type": f["type"], "label": f.get("label", f["name"]), "required": f.get("required", True)}
                for f in config.get("pathology_specific_fields", [])
            ],
        })
    return {"pathologies": result}


@app.get(
    "/api/v1/scdp/pathologies/{pathology_id}/schema",
    tags=["SCDP"],
    summary="JSON Schema des champs clinical_data pour une pathologie",
)
async def get_pathology_schema(
    pathology_id: str,
    current_user: database.User = Depends(get_current_user),
):
    schema = core.get_pathology_schema(pathology_id)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Pathologie '{pathology_id}' inconnue.")
    return schema


@app.get(
    "/api/v1/scdp/pathologies/{pathology_id}/params",
    tags=["SCDP"],
    summary="Paramètres courants d'une pathologie (lecture seule)",
)
async def get_pathology_params(
    pathology_id: str,
    current_user: database.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    # Admin uniquement
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs.")

    params = database.get_active_params(db, pathology_id)
    if params is None:
        raise HTTPException(status_code=404, detail=f"Aucun paramètre actif pour '{pathology_id}'.")

    database.write_audit_log(
        db=db,
        action="scdp.params.read",
        actor_id=current_user.id,
        payload={"pathology_id": pathology_id},
    )

    return {
        "pathology_id": params.pathology_id,
        "version": params.version,
        "sha256": params.sha256,
        "is_active": params.is_active,
        "created_at": params.created_at.isoformat(),
        "params": params.params_json,
    }


# ─────────────────────────────────────────────
# ENDPOINTS REPORTS
# ─────────────────────────────────────────────

@app.get(
    "/api/v1/reports",
    response_model=PaginatedReports,
    tags=["Reports"],
    summary="Liste paginée des rapports (médecin : les siens ; admin : tous)",
)
async def list_reports(
    page: int = 1,
    page_size: int = 20,
    pathology_id: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: database.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    if current_user.role == "admin":
        reports, total = database.get_all_reports(
            db,
            page=page,
            page_size=page_size,
            pathology_filter=pathology_id,
            user_filter=user_id,
        )
    else:
        reports, total = database.get_reports_by_user(
            db,
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            pathology_filter=pathology_id,
        )

    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaginatedReports(
        items=[_report_to_summary(r) for r in reports],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.get(
    "/api/v1/reports/{report_id}",
    tags=["Reports"],
    summary="Détail complet d'un rapport",
)
async def get_report(
    report_id: str,
    current_user: database.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    report = database.get_report_by_id(db, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Rapport introuvable.")

    # Médecin ne peut accéder qu'à ses propres rapports
    if current_user.role != "admin" and report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accès non autorisé à ce rapport.")

    return {
        "id": report.id,
        "user_id": report.user_id,
        "pathology_id": report.pathology_id,
        "pdf_available": report.pdf_path is not None,
        "created_at": report.created_at.isoformat(),
        **report.report_json,
    }


@app.get(
    "/api/v1/reports/{report_id}/pdf",
    tags=["Reports"],
    summary="Téléchargement du rapport PDF",
)
async def download_report_pdf(
    report_id: str,
    current_user: database.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    report = database.get_report_by_id(db, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Rapport introuvable.")

    if current_user.role != "admin" and report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Accès non autorisé.")

    if not report.pdf_path:
        raise HTTPException(status_code=404, detail="PDF non disponible pour ce rapport.")

    if not os.path.exists(report.pdf_path):
        raise HTTPException(status_code=404, detail="Fichier PDF introuvable sur le serveur.")

    return FileResponse(
        path=report.pdf_path,
        media_type="application/pdf",
        filename=f"SCDP_{report.pathology_id}_{report_id[:8]}.pdf",
    )


# ─────────────────────────────────────────────
# ENDPOINTS ADMIN
# ─────────────────────────────────────────────

@app.get(
    "/api/v1/admin/audit-logs",
    response_model=PaginatedAuditLogs,
    tags=["Admin"],
    summary="Audit logs (admin uniquement)",
)
async def list_audit_logs(
    page: int = 1,
    page_size: int = 50,
    actor_id: Optional[str] = None,
    action: Optional[str] = None,
    current_user: database.User = Depends(require_admin),
    db: Session = Depends(database.get_db),
):
    logs, total = database.get_audit_logs(
        db,
        page=page,
        page_size=page_size,
        actor_filter=actor_id,
        action_filter=action,
    )
    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaginatedAuditLogs(
        items=[
            AuditLogResponse(
                id=log.id,
                actor_id=log.actor_id,
                action=log.action,
                payload=log.payload,
                ip_address=log.ip_address,
                created_at=log.created_at,
            )
            for log in logs
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.get(
    "/api/v1/admin/stats",
    tags=["Admin"],
    summary="Statistiques agrégées (admin uniquement)",
)
async def admin_stats(
    current_user: database.User = Depends(require_admin),
    db: Session = Depends(database.get_db),
):
    from sqlalchemy import func

    total_users = db.query(func.count(database.User.id)).scalar()
    total_reports = db.query(func.count(database.Report.id)).scalar()

    # Répartition par pathologie
    pathology_counts = (
        db.query(database.Report.pathology_id, func.count(database.Report.id))
        .group_by(database.Report.pathology_id)
        .all()
    )

    # Répartition par décision
    decision_counts: dict[str, int] = {}
    reports_all = db.query(database.Report).all()
    for r in reports_all:
        dc = r.report_json.get("decision_class", "INCERTAIN") if r.report_json else "INCERTAIN"
        decision_counts[dc] = decision_counts.get(dc, 0) + 1

    return {
        "total_users": total_users,
        "total_reports": total_reports,
        "by_pathology": {pid: cnt for pid, cnt in pathology_counts},
        "by_decision": decision_counts,
    }


# ─────────────────────────────────────────────
# ENDPOINTS SYSTÈME
# ─────────────────────────────────────────────

@app.get("/health", tags=["Système"], summary="Healthcheck")
async def health(db: Session = Depends(database.get_db)):
    try:
        db.execute(database.text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    uptime = round(time.time() - APP_START_TIME, 2)
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "version": APP_VERSION,
        "uptime_seconds": uptime,
        "database": db_status,
        "pathologies_loaded": len(core.list_pathology_ids()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics", tags=["Système"], summary="Métriques Prometheus")
async def metrics(current_user: database.User = Depends(require_admin)):
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)