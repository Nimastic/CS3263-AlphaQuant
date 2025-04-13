"""
API Main Module.

This module sets up the FastAPI application and includes the API routes.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AlphaQuant API",
    description="API for AlphaQuant AI-Powered Investment Advisor",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include API routes
from .routes import router
app.include_router(router)

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Execute tasks when the application starts."""
    logger.info("Starting AlphaQuant API...")
    
    # Log environment configuration
    api_port = os.getenv("PORT", "8000")
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"API Port: {api_port}")
    logger.info(f"Debug Mode: {debug_mode}")

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute tasks when the application shuts down."""
    logger.info("Shutting down AlphaQuant API...")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that provides basic information about the API."""
    return {
        "name": "AlphaQuant API",
        "version": "1.0.0",
        "description": "AI-Powered Investment Advisor",
        "docs": "/docs",
        "status": "operational"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring the API."""
    return {"status": "ok", "message": "AlphaQuant API is operational"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Get host from environment or use default
    host = os.getenv("HOST", "0.0.0.0")
    
    # Get debug mode from environment or use default
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    # Run the application
    uvicorn.run("src.api.main:app", host=host, port=port, reload=debug) 