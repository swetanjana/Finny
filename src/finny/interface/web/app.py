"""FastAPI application factory for Finny web interface."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from finny.interface.web.routes import router

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")

    app = FastAPI(
        title="Finny — Financial Data Agent",
        description="Agentic Q&A over Apple SEC filings with RBAC and streaming tool-call traces.",
        version="0.1.0",
    )

    app.include_router(router)

    # Mount static files last so API routes take precedence
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app
