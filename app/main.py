# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routers.chat import router as chat_router
from .core.logging_config import get_logger
from .api.services.chat_engine import init_model

# Initialize logger
logger = get_logger("main")

# Create FastAPI app
app = FastAPI(
    title="Chat API",
    description="A simple chat API using Hugging Face models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the model when the application starts."""
    logger.info("Starting up application...")
    try:
        init_model()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that provides basic API information."""
    return {
        "name": "Chat API",
        "version": "1.0.0",
        "description": "A simple chat API using Hugging Face models",
        "documentation": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
