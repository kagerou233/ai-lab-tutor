"""Main FastAPI application module.

This module initializes the FastAPI application and registers all route handlers.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logging_config import setup_logging
from config import (
    CORS_ALLOWED_ORIGINS,
    API_HOST,
    API_PORT,
    DATA_DIR,
)
from api.routes import auth, profile, session, interaction, adapter, class_route
from api.routes import settings
from api.routes import custom_skill
from utils.model_manager import check_and_download_models
from utils.skills import warmup_embeddings

# Configure tiktoken to use local cache (avoid network download issues in China)
# This prevents SSL errors when downloading encoding files from Azure blob storage
tiktoken_cache_dir = DATA_DIR / "tiktoken_cache"
tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

# Setup logging
setup_logging()

# Initialize FastAPI application
app = FastAPI(
    title="Interactive Tutor API",
    description="Backend API service for interactive AI tutoring system.",
    version="2.0.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: Model checking is done synchronously before starting the server
# See the __main__ block below for the actual model check

# Register route handlers
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(session.router)
app.include_router(interaction.router)
app.include_router(adapter.router)
app.include_router(class_route.router)
app.include_router(custom_skill.router)
app.include_router(settings.router)


@app.on_event("startup")
def startup_tasks() -> None:
    """Warm up shared embeddings and pre-cache tiktoken encodings."""
    warmup_embeddings()
    # Pre-cache tiktoken encoding to avoid SSL errors during requests
    _preload_tiktoken_encodings()


def _preload_tiktoken_encodings() -> None:
    """Pre-load tiktoken encodings to cache them locally.

    This prevents SSL/connection errors when tiktoken tries to download
    encoding files from Azure blob storage during the first request.
    """
    try:
        import tiktoken
        # Pre-load common encodings used by the system
        tiktoken.get_encoding("cl100k_base")  # GPT-4/GPT-3.5-turbo
        print("✓ tiktoken encoding cached successfully")
    except Exception as e:
        print(f"⚠ Failed to pre-cache tiktoken encoding: {e}")
        print("  Will attempt to download during first request (may fail in China)")


@app.get("/", summary="API 根路径", tags=["Info"])
def root() -> dict:
    """API 根路径，返回 API 信息和文档链接。

    Returns:
        Dictionary with API information and documentation links.
    """
    return {
        "name": "Interactive Tutor API",
        "version": "2.0.0",
        "description": "Backend API service for interactive AI tutoring system.",
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
        },
        "health": "/api/health",
    }


@app.get("/api/health", summary="健康检查", tags=["Health"])
def health() -> dict:
    """Health check endpoint.

    Returns:
        Dictionary with status "ok".
    """
    return {"status": "ok"}


# --- Startup code for direct execution ---
if __name__ == "__main__":
    import uvicorn

    # Check and download models BEFORE starting the server
    # This ensures models are ready before FastAPI starts and avoids
    # file change detection interrupting the download process
    print("\n" + "=" * 60)
    print("🚀 启动模型检查...")
    print("=" * 60)
    
    all_successful, downloaded, failed = check_and_download_models()
    
    if not all_successful:
        print("\n⚠️  警告: 部分模型下载失败，某些功能可能无法正常使用")
        print("   请检查网络连接或手动下载失败的模型")
        print("   服务器将继续启动，但相关功能可能不可用")
    elif downloaded:
        print("\n✅ 所有模型已准备就绪")
    
    
    print("=" * 60 + "\n")
    
    # Now start the FastAPI server after models are ready
    print("🚀 启动 Socratic Agent API 服务器...")
    server_url = f"http://{API_HOST}:{API_PORT}"
    print(f"🌐 服务地址(后端服务): {server_url}")
    print(f"📚 API 文档: {server_url}/docs")
    print()

    # reload=True enables auto-reload on code changes
    # Models are already downloaded, so file changes won't interrupt downloads
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
