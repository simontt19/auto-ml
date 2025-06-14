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

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import jwt
import hashlib

# Import our modules with proper path handling
try:
    from auto_ml.core.user_management import UserManager, User, Project
    from auto_ml.models.persistence import ModelPersistence
    from auto_ml.features.engineering import StandardFeatureEngineering
    from auto_ml.models.persistence.model_registry import ModelRegistry
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

# JWT Configuration
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"

# Simple password storage (in production, use proper password hashing)
DEMO_PASSWORDS = {
    "admin": "admin123",
    "testuser": "test123"
}

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    user: Dict[str, Any]
    token: str

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
model_registry = ModelRegistry()

def create_access_token(data: dict):
    """Create JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.PyJWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from token."""
    token = credentials.credentials
    username = verify_token(token)
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = user_manager.get_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Auto ML API is running!", "version": "1.0.0"}

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint."""
    # Check if user exists
    user = user_manager.get_user(request.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check password (simple demo implementation)
    if request.password != DEMO_PASSWORDS.get(request.username):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token = create_access_token({"sub": request.username})
    
    # Update last login
    user_manager.update_user(request.username, last_login=datetime.now().isoformat())
    
    return LoginResponse(
        user={
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "projects": user.projects
        },
        token=token
    )

@app.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role.value,
        "projects": current_user.projects
    }

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
        return {"users": [{"username": u.username, "email": u.email, "role": u.role.value} for u in users]}
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/projects")
async def list_projects():
    """List all projects."""
    try:
        projects = user_manager.list_all_projects()
        return {"projects": [{"project_id": p.project_id, "name": p.name, "owner": p.owner, "description": p.description} for p in projects]}
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, current_user: User = Depends(get_current_user)):
    """Make a prediction."""
    try:
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

@app.get("/models/registry")
async def list_registered_models():
    """List all registered models and their metadata."""
    return {"models": model_registry.list_models()}

@app.get("/models/registry/{model_id}")
async def get_registered_model(model_id: str):
    """Get metadata for a specific model."""
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.post("/models/registry")
async def register_model(metadata: dict = Body(...)):
    """Register a new model with metadata."""
    model_registry.register_model(metadata)
    return {"status": "registered", "model_id": metadata.get("model_id")}

@app.put("/models/registry/{model_id}")
async def update_registered_model(model_id: str, updates: dict = Body(...)):
    """Update metadata for a specific model."""
    updated = model_registry.update_model(model_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Model not found or not updated")
    return {"status": "updated", "model_id": model_id}

@app.delete("/models/registry/{model_id}")
async def delete_registered_model(model_id: str):
    """Delete a model from the registry."""
    deleted = model_registry.delete_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found or not deleted")
    return {"status": "deleted", "model_id": model_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 