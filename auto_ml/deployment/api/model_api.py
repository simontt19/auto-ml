"""
Production Model API with Multi-User Support
FastAPI-based REST API for model serving in production with user authentication and project isolation.
"""

import logging
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import joblib

# Use relative imports that work with the current structure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.persistence import ModelPersistence
from features.engineering import StandardFeatureEngineering
from core.exceptions import ModelPersistenceError
from core.user_management import UserManager, User, Project

logger = logging.getLogger(__name__)

# Security scheme for API authentication
security = HTTPBearer()

class PredictionRequest(BaseModel):
    """Request model for prediction API."""
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    model_name: Optional[str] = Field(None, description="Specific model name to use")
    version: Optional[str] = Field(None, description="Specific model version to use")
    project_id: Optional[str] = Field(None, description="Project ID for model context")

class PredictionResponse(BaseModel):
    """Response model for prediction API."""
    prediction: float = Field(..., description="Model prediction")
    probability: float = Field(..., description="Prediction probability")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    user_info: Dict[str, Any] = Field(..., description="User information")
    project_info: Dict[str, Any] = Field(..., description="Project information")
    timestamp: str = Field(..., description="Prediction timestamp")

class ModelInfoResponse(BaseModel):
    """Response model for model information API."""
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type")
    training_results: Dict[str, Any] = Field(..., description="Training performance")
    feature_names: List[str] = Field(..., description="Required feature names")
    model_size_mb: float = Field(..., description="Model file size in MB")
    created_at: str = Field(..., description="Model creation timestamp")
    user: str = Field(..., description="Model owner")
    project: str = Field(..., description="Project ID")

class HealthResponse(BaseModel):
    """Response model for health check API."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    models_loaded: int = Field(..., description="Number of models loaded")
    active_models: List[str] = Field(..., description="List of active model names")
    users_online: int = Field(..., description="Number of active users")
    projects_active: int = Field(..., description="Number of active projects")

class UserContext:
    """User context for API requests."""
    def __init__(self, user: User, project: Optional[Project] = None):
        self.user = user
        self.project = project

class ModelAPI:
    """
    Production-ready model serving API using FastAPI with multi-user support.
    
    This class provides:
    - REST API endpoints for model predictions with user authentication
    - Project-specific model isolation and access control
    - Model health monitoring
    - Automatic model loading and caching
    - Input validation and error handling
    - Performance metrics tracking
    
    Attributes:
        app (FastAPI): FastAPI application instance
        user_manager (UserManager): User management system
        model_persistence (ModelPersistence): Model persistence manager
        feature_engineering (StandardFeatureEngineering): Feature engineering pipeline
        loaded_models (Dict): Cache of loaded models
        model_metadata (Dict): Cache of model metadata
    """
    
    def __init__(self, models_base_dir: str = "projects", host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the Model API with multi-user support.
        
        Args:
            models_base_dir (str): Base directory containing user project models
            host (str): Host address for the API server
            port (int): Port number for the API server
        """
        self.models_base_dir = Path(models_base_dir)
        self.host = host
        self.port = port
        
        # Initialize user management
        self.user_manager = UserManager()
        
        # Initialize components
        self.feature_engineering = StandardFeatureEngineering()
        
        # Model cache (user -> project -> model_name -> model)
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Auto ML Model API",
            description="Production-ready API for serving machine learning models with multi-user support",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info("Multi-user Model API initialized successfully")
    
    async def _get_user_context(self, 
                               credentials: HTTPAuthorizationCredentials = Depends(security),
                               project_id: Optional[str] = None) -> UserContext:
        """
        Get user context from authentication token.
        
        Args:
            credentials: HTTP authorization credentials
            project_id: Optional project ID for context
            
        Returns:
            UserContext: User and project context
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # For now, use simple token-based auth (in production, use JWT)
            token = credentials.credentials
            
            # Simple token validation (username:token format)
            if ':' not in token:
                raise HTTPException(status_code=401, detail="Invalid token format")
            
            username, token_value = token.split(':', 1)
            
            # Get user
            user = self.user_manager.get_user(username)
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            
            # Validate token (simple check for now)
            if not self._validate_token(username, token_value):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Get project context if specified
            project = None
            if project_id:
                project = self.user_manager.get_project(project_id)
                if project is None:
                    raise HTTPException(status_code=404, detail="Project not found")
                
                # Check user access to project
                if not self.user_manager.check_permission(username, project_id, "read"):
                    raise HTTPException(status_code=403, detail="Access denied to project")
            
            return UserContext(user, project)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    def _validate_token(self, username: str, token: str) -> bool:
        """
        Validate user token (simplified for development).
        
        Args:
            username: Username
            token: Token to validate
            
        Returns:
            bool: True if token is valid
        """
        # Simple token validation (in production, use proper JWT validation)
        # For development, accept any non-empty token
        return len(token) > 0
    
    def _get_project_models_dir(self, user: User, project: Project) -> Path:
        """Get project-specific models directory."""
        return self.models_base_dir / user.username / project.project_id / "models"
    
    def _get_model_persistence(self, user: User, project: Project) -> ModelPersistence:
        """Get project-specific model persistence instance."""
        models_dir = self._get_project_models_dir(user, project)
        return ModelPersistence(str(models_dir))
    
    def _register_routes(self):
        """Register API routes with authentication."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Auto ML Model API with Multi-User Support",
                "version": "2.0.0",
                "docs": "/docs",
                "health": "/health",
                "authentication": "Bearer token required (username:token format)"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            # Count active users and projects
            users_online = len(self.user_manager.users)
            projects_active = len(self.user_manager.projects)
            
            # Count total loaded models across all users/projects
            total_models = sum(len(project_models) for user_models in self.loaded_models.values() 
                             for project_models in user_models.values())
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                models_loaded=total_models,
                active_models=[],  # Would need to collect all model names
                users_online=users_online,
                projects_active=projects_active
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, 
                         user_context: UserContext = Depends(self._get_user_context)):
            """Make predictions using the specified model with user/project context."""
            try:
                # Use project from request or user context
                project = user_context.project
                if request.project_id and not project:
                    project = self.user_manager.get_project(request.project_id)
                    if not project or not self.user_manager.check_permission(
                        user_context.user.username, request.project_id, "read"):
                        raise HTTPException(status_code=403, detail="Access denied to project")
                
                if not project:
                    raise HTTPException(status_code=400, detail="Project ID required")
                
                # Get project-specific model persistence
                model_persistence = self._get_model_persistence(user_context.user, project)
                
                # Get model name and version
                model_name = request.model_name or self._get_default_model_name(user_context.user, project)
                version = request.version
                
                # Load model if not cached
                cache_key = f"{user_context.user.username}:{project.project_id}:{model_name}"
                if cache_key not in self.loaded_models:
                    await self._load_model(user_context.user, project, model_name, version)
                
                # Prepare features
                features_df = self._prepare_features(request.features, cache_key)
                
                # Make prediction
                model = self.loaded_models[cache_key]
                metadata = self.model_metadata[cache_key]
                
                # Get prediction and probability
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_df)[0]
                    prediction = model.predict(features_df)[0]
                    probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                else:
                    prediction = model.predict(features_df)[0]
                    probability = 0.5  # Default probability for models without predict_proba
                
                return PredictionResponse(
                    prediction=float(prediction),
                    probability=float(probability),
                    model_info={
                        "model_name": model_name,
                        "version": metadata["version"],
                        "model_type": metadata["model_type"]
                    },
                    user_info={
                        "username": user_context.user.username,
                        "role": user_context.user.role.value
                    },
                    project_info={
                        "project_id": project.project_id,
                        "name": project.name
                    },
                    timestamp=datetime.now().isoformat()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", response_model=Dict[str, List[ModelInfoResponse]])
        async def list_models(user_context: UserContext = Depends(self._get_user_context),
                            project_id: Optional[str] = None):
            """List available models for the authenticated user."""
            try:
                # Determine which projects to list models for
                projects_to_check = []
                
                if project_id:
                    # Specific project
                    project = self.user_manager.get_project(project_id)
                    if not project or not self.user_manager.check_permission(
                        user_context.user.username, project_id, "read"):
                        raise HTTPException(status_code=403, detail="Access denied to project")
                    projects_to_check = [project]
                else:
                    # All user's projects
                    projects_to_check = self.user_manager.list_user_projects(user_context.user.username)
                
                result = {}
                
                for project in projects_to_check:
                    try:
                        model_persistence = self._get_model_persistence(user_context.user, project)
                        models = model_persistence.list_models()
                        
                        project_models = []
                        for model_name, versions in models.items():
                            for version_info in versions:
                                try:
                                    metadata = model_persistence.get_model_info(
                                        model_name, version_info["version"]
                                    )
                                    project_models.append(ModelInfoResponse(
                                        model_name=model_name,
                                        version=metadata["version"],
                                        model_type=metadata["model_type"],
                                        training_results=metadata["training_results"],
                                        feature_names=metadata["feature_names"],
                                        model_size_mb=metadata["model_size_mb"],
                                        created_at=metadata["timestamp"],
                                        user=user_context.user.username,
                                        project=project.project_id
                                    ))
                                except Exception as e:
                                    logger.warning(f"Failed to get info for {model_name} {version_info['version']}: {e}")
                        
                        result[project.project_id] = project_models
                        
                    except Exception as e:
                        logger.warning(f"Failed to list models for project {project.project_id}: {e}")
                        result[project.project_id] = []
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_name}", response_model=ModelInfoResponse)
        async def get_model_info(model_name: str, 
                               version: Optional[str] = None,
                               project_id: Optional[str] = None,
                               user_context: UserContext = Depends(self._get_user_context)):
            """Get information about a specific model."""
            try:
                if not project_id:
                    raise HTTPException(status_code=400, detail="Project ID required")
                
                project = self.user_manager.get_project(project_id)
                if not project or not self.user_manager.check_permission(
                    user_context.user.username, project_id, "read"):
                    raise HTTPException(status_code=403, detail="Access denied to project")
                
                model_persistence = self._get_model_persistence(user_context.user, project)
                metadata = model_persistence.get_model_info(model_name, version)
                
                return ModelInfoResponse(
                    model_name=model_name,
                    version=metadata["version"],
                    model_type=metadata["model_type"],
                    training_results=metadata["training_results"],
                    feature_names=metadata["feature_names"],
                    model_size_mb=metadata["model_size_mb"],
                    created_at=metadata["timestamp"],
                    user=user_context.user.username,
                    project=project.project_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                raise HTTPException(status_code=404, detail=f"Model not found: {e}")
        
        @self.app.post("/models/{model_name}/load")
        async def load_model(model_name: str, 
                           version: Optional[str] = None,
                           project_id: Optional[str] = None,
                           background_tasks: BackgroundTasks = None,
                           user_context: UserContext = Depends(self._get_user_context)):
            """Load a specific model into memory."""
            try:
                if not project_id:
                    raise HTTPException(status_code=400, detail="Project ID required")
                
                project = self.user_manager.get_project(project_id)
                if not project or not self.user_manager.check_permission(
                    user_context.user.username, project_id, "read"):
                    raise HTTPException(status_code=403, detail="Access denied to project")
                
                await self._load_model(user_context.user, project, model_name, version)
                return {"message": f"Model {model_name} loaded successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/models/{model_name}/unload")
        async def unload_model(model_name: str,
                             project_id: Optional[str] = None,
                             user_context: UserContext = Depends(self._get_user_context)):
            """Unload a model from memory."""
            try:
                if not project_id:
                    raise HTTPException(status_code=400, detail="Project ID required")
                
                project = self.user_manager.get_project(project_id)
                if not project or not self.user_manager.check_permission(
                    user_context.user.username, project_id, "read"):
                    raise HTTPException(status_code=403, detail="Access denied to project")
                
                cache_key = f"{user_context.user.username}:{project.project_id}:{model_name}"
                if cache_key in self.loaded_models:
                    del self.loaded_models[cache_key]
                    del self.model_metadata[cache_key]
                    return {"message": f"Model {model_name} unloaded successfully"}
                else:
                    return {"message": f"Model {model_name} was not loaded"}
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _load_model(self, user: User, project: Project, model_name: str, version: Optional[str] = None):
        """Load a model into memory cache."""
        try:
            model_persistence = self._get_model_persistence(user, project)
            model, metadata = model_persistence.load_model(model_name, version)
            
            cache_key = f"{user.username}:{project.project_id}:{model_name}"
            self.loaded_models[cache_key] = model
            self.model_metadata[cache_key] = metadata
            
            logger.info(f"Model {model_name} loaded for user {user.username} and project {project.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelPersistenceError(f"Failed to load model {model_name}: {e}")
    
    def _get_default_model_name(self, user: User, project: Project) -> str:
        """Get the default model name for a user/project."""
        try:
            model_persistence = self._get_model_persistence(user, project)
            models = model_persistence.list_models()
            if models:
                return list(models.keys())[0]
            else:
                raise HTTPException(status_code=404, detail="No models available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting default model: {e}")
    
    def _prepare_features(self, features: Dict[str, Any], cache_key: str) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            metadata = self.model_metadata[cache_key]
            feature_names = metadata["feature_names"]
            
            # Create DataFrame with features
            features_df = pd.DataFrame([features])
            
            # Ensure all required features are present
            missing_features = set(feature_names) - set(features_df.columns)
            if missing_features:
                # Add missing features with default values
                for feature in missing_features:
                    features_df[feature] = 0
            
            # Select only the required features in the correct order
            features_df = features_df[feature_names]
            
            # Handle missing values
            features_df = features_df.fillna(0)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid features: {e}")
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """Run the API server."""
        host = host or self.host
        port = port or self.port
        
        logger.info(f"Starting Multi-User Auto ML Model API on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app 