# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .api.services.chat_engine import *
from .api.routers.chat import router as chat_router
from .core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up…")
    init_model()
    try:
        yield
    finally:
        logger.info("Shutting down…")
        shutdown_model()

app = FastAPI(title="chat-app", version="0.1", lifespan=lifespan)
app.include_router(chat_router, prefix="/api")
