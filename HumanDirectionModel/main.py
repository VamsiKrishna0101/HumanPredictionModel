from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(title="Human Direction Detection API", version="0.115.2")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(api_router, prefix="/v1")
    return app

app = create_app()
