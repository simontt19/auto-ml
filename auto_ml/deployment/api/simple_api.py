#!/usr/bin/env python3
"""
Simple FastAPI server for testing the Auto ML framework.
"""

import sys
import os
from pathlib import Path

# Add the auto_ml directory to Python path
current_dir = Path(__file__).parent
auto_ml_dir = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(auto_ml_dir))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Import our modules with proper path handling
try:
    from auto_ml.core.user_management import UserManager, User, Project
    from auto_ml.models.persistence import ModelPersistence
    from auto_ml.features.engineering import StandardFeatureEngineering
except ImportError as e:
    # Fallback for deployment environment
    logging.warning(f"Import error: {e}. Using mock implementations.")
    
    # Mock classes for deployment
    class UserManager:
        def __init__(self):
            self.users = []
            self.projects = []
        
        def list_users(self):
            return [{"username": "testuser", "email": "test@example.com"}]
        
        def list_projects(self):
            return [{"project_id": "test_project", "name": "Test Project", "owner": "testuser"}]
    
    class StandardFeatureEngineering:
        def __init__(self):
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    users_online: int
    projects_active: int

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_name: Optional[str] = None
    project_id: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_info: Dict[str, Any]
    timestamp: str

# Create FastAPI app
app = FastAPI(
    title="Auto ML API",
    description="Simple API for Auto ML framework testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
user_manager = UserManager()
feature_engineering = StandardFeatureEngineering()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Auto ML API is running!", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=0,
        users_online=1,
        projects_active=2
    )

@app.get("/users")
async def list_users():
    """List all users."""
    try:
        users = user_manager.list_users()
        return {"users": users}
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/projects")
async def list_projects():
    """List all projects."""
    try:
        projects = user_manager.list_projects()
        return {"projects": projects}
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Make a prediction."""
    try:
        # Simple token validation
        token = credentials.credentials
        if not token or len(token) < 3:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Mock prediction for testing
        prediction = 0.85
        probability = 0.92
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_info={
                "model_name": request.model_name or "default_model",
                "version": "1.0.0",
                "features_used": list(request.features.keys())
            },
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/docs")
async def get_docs():
    """API documentation."""
    return {"message": "API documentation available at /docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 