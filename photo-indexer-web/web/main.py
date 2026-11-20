import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.api import assets, faces, immich, measurements, performers, system, tags
from web.auth import create_access_token, get_current_user, get_current_user_optional, verify_password
from web.config import get_settings
from web.db import close_db, get_db
from web.jobs.manager import close_redis, get_job_manager
from web.jobs.routes import router as jobs_router
from web.media import serve_media_file
from web.redis import close_redis as close_redis_conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    settings = get_settings()

    # Initialize DB
    db = await get_db()

    # Initialize job manager
    await get_job_manager()

    yield

    # Shutdown
    await close_db()
    await close_redis()
    await close_redis_conn()


app = FastAPI(
    title="Photo Indexer Web UI",
    description="Companion web UI for photo-indexer CLI",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Templates
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_path)

# Include API routers
app.include_router(performers.router)
app.include_router(faces.router)
app.include_router(measurements.router)
app.include_router(tags.router)
app.include_router(assets.router)
app.include_router(immich.router)
app.include_router(system.router)
app.include_router(jobs_router)


# Auth routes
from fastapi import Form
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.post("/api/v1/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint - returns JWT token."""
    settings = get_settings()

    # Verify password against stored hash
    if not verify_password(request.password, settings.web_password):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if request.username != settings.web_username:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": request.username})
    return LoginResponse(access_token=access_token)

# Auth me endpoint
# Auth me endpoint
@app.get("/api/v1/auth/me")
async def get_me(user=Depends(get_current_user_optional)):
    """Get current user info."""
    if user:
        return {"username": user.username, "authenticated": True}
    return {"authenticated": False}




# Health check (no auth required)
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_connected = False
    immich_connected = False
    redis_connected = False
    ollama_connected = False

    try:
        db = await get_db()
        await db.get_stats()
        db_connected = True
    except Exception:
        pass

    try:
        from web.immich import get_immich_client
        immich = get_immich_client()
        immich_connected = await immich.health_check()
    except Exception:
        pass

    try:
        from web.redis import get_redis
        redis_client = await get_redis()
        await redis_client.ping()
        redis_connected = True
    except Exception:
        pass

    try:
        import httpx
        settings = get_settings()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_connected = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if db_connected else "degraded",
        "db_connected": db_connected,
        "immich_connected": immich_connected,
        "redis_connected": redis_connected,
        "ollama_connected": ollama_connected,
    }


# Prometheus metrics
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Frontend routes - serve HTML templates
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse(request, "login.html")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user=Depends(get_current_user_optional)):
    """Root - redirect to performers if authenticated, else login."""
    if user:
        return RedirectResponse(url="/performers")
    return RedirectResponse(url="/login")


@app.get("/performers", response_class=HTMLResponse)
async def performers_page(request: Request, user=Depends(get_current_user)):
    """Performers list page."""
    return templates.TemplateResponse(request, "performers.html")


@app.get("/faces", response_class=HTMLResponse)
async def faces_page(request: Request, user=Depends(get_current_user)):
    """Faces review page."""
    return templates.TemplateResponse(request, "faces.html")


@app.get("/measurements", response_class=HTMLResponse)
async def measurements_page(request: Request, user=Depends(get_current_user)):
    """Measurements browser page."""
    return templates.TemplateResponse(request, "measurements.html")


@app.get("/tags", response_class=HTMLResponse)
async def tags_page(request: Request, user=Depends(get_current_user)):
    """Tags management page."""
    return templates.TemplateResponse(request, "tags.html")


@app.get("/assets", response_class=HTMLResponse)
async def assets_page(request: Request, user=Depends(get_current_user)):
    """Assets browser page."""
    return templates.TemplateResponse(request, "assets.html")


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request, user=Depends(get_current_user)):
    """Jobs dashboard page."""
    return templates.TemplateResponse(request, "jobs.html")


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user=Depends(get_current_user)):
    """Settings page."""
    return templates.TemplateResponse(request, "settings.html")


# 404 handler
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse({"detail": "Not found"}, status_code=404)


# 500 handler
@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


def main():
    """Entry point for running with uvicorn directly."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()