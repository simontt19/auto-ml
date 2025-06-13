"""
Production Model API
FastAPI-based REST API for model serving in production.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

from ...models.persistence import ModelPersistence
from ...features.engineering import StandardFeatureEngineering
from ...core.exceptions import ModelPersistenceError

logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Request model for prediction API."""
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    model_name: Optional[str] = Field(None, description="Specific model name to use")
    version: Optional[str] = Field(None, description="Specific model version to use")

class PredictionResponse(BaseModel):
    """Response model for prediction API."""
    prediction: float = Field(..., description="Model prediction")
    probability: float = Field(..., description="Prediction probability")
    model_info: Dict[str, Any] = Field(..., description="Model information")
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

class HealthResponse(BaseModel):
    """Response model for health check API."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    models_loaded: int = Field(..., description="Number of models loaded")
    active_models: List[str] = Field(..., description="List of active model names")

class ModelAPI:
    """
    Production-ready model serving API using FastAPI.
    
    This class provides:
    - REST API endpoints for model predictions
    - Model health monitoring
    - Automatic model loading and caching
    - Input validation and error handling
    - Performance metrics tracking
    
    Attributes:
        app (FastAPI): FastAPI application instance
        model_persistence (ModelPersistence): Model persistence manager
        feature_engineering (StandardFeatureEngineering): Feature engineering pipeline
        loaded_models (Dict): Cache of loaded models
        model_metadata (Dict): Cache of model metadata
    """
    
    def __init__(self, models_dir: str = "models", host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the Model API.
        
        Args:
            models_dir (str): Directory containing saved models
            host (str): Host address for the API server
            port (int): Port number for the API server
        """
        self.models_dir = models_dir
        self.host = host
        self.port = port
        
        # Initialize components
        self.model_persistence = ModelPersistence(models_dir)
        self.feature_engineering = StandardFeatureEngineering()
        
        # Model cache
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Auto ML Model API",
            description="Production-ready API for serving machine learning models",
            version="1.0.0",
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
        
        # Load default models
        self._load_default_models()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Auto ML Model API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                models_loaded=len(self.loaded_models),
                active_models=list(self.loaded_models.keys())
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make predictions using the specified model."""
            try:
                # Get model name and version
                model_name = request.model_name or self._get_default_model_name()
                version = request.version
                
                # Load model if not cached
                if model_name not in self.loaded_models:
                    await self._load_model(model_name, version)
                
                # Prepare features
                features_df = self._prepare_features(request.features, model_name)
                
                # Make prediction
                model = self.loaded_models[model_name]
                metadata = self.model_metadata[model_name]
                
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
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", response_model=Dict[str, List[ModelInfoResponse]])
        async def list_models():
            """List all available models."""
            try:
                models = self.model_persistence.list_models()
                result = {}
                
                for model_name, versions in models.items():
                    model_infos = []
                    for version_info in versions:
                        try:
                            metadata = self.model_persistence.get_model_info(
                                model_name, version_info["version"]
                            )
                            model_infos.append(ModelInfoResponse(
                                model_name=model_name,
                                version=metadata["version"],
                                model_type=metadata["model_type"],
                                training_results=metadata["training_results"],
                                feature_names=metadata["feature_names"],
                                model_size_mb=metadata["model_size_mb"],
                                created_at=metadata["timestamp"]
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to get info for {model_name} {version_info['version']}: {e}")
                    
                    result[model_name] = model_infos
                
                return result
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_name}", response_model=ModelInfoResponse)
        async def get_model_info(model_name: str, version: Optional[str] = None):
            """Get information about a specific model."""
            try:
                metadata = self.model_persistence.get_model_info(model_name, version)
                return ModelInfoResponse(
                    model_name=model_name,
                    version=metadata["version"],
                    model_type=metadata["model_type"],
                    training_results=metadata["training_results"],
                    feature_names=metadata["feature_names"],
                    model_size_mb=metadata["model_size_mb"],
                    created_at=metadata["timestamp"]
                )
                
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                raise HTTPException(status_code=404, detail=f"Model not found: {e}")
        
        @self.app.post("/models/{model_name}/load")
        async def load_model(model_name: str, version: Optional[str] = None, background_tasks: BackgroundTasks = None):
            """Load a specific model into memory."""
            try:
                await self._load_model(model_name, version)
                return {"message": f"Model {model_name} loaded successfully"}
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/models/{model_name}/unload")
        async def unload_model(model_name: str):
            """Unload a model from memory."""
            try:
                if model_name in self.loaded_models:
                    del self.loaded_models[model_name]
                    del self.model_metadata[model_name]
                    return {"message": f"Model {model_name} unloaded successfully"}
                else:
                    return {"message": f"Model {model_name} was not loaded"}
                    
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _load_model(self, model_name: str, version: Optional[str] = None):
        """Load a model into memory cache."""
        try:
            model, metadata = self.model_persistence.load_model(model_name, version)
            self.loaded_models[model_name] = model
            self.model_metadata[model_name] = metadata
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelPersistenceError(f"Failed to load model {model_name}: {e}")
    
    def _load_default_models(self):
        """Load default models on startup."""
        try:
            models = self.model_persistence.list_models()
            if models:
                # Load the first available model as default
                default_model_name = list(models.keys())[0]
                logger.info(f"Loading default model: {default_model_name}")
                # Note: This is synchronous, but in production you might want async loading
                model, metadata = self.model_persistence.load_model(default_model_name)
                self.loaded_models[default_model_name] = model
                self.model_metadata[default_model_name] = metadata
                
        except Exception as e:
            logger.warning(f"Failed to load default models: {e}")
    
    def _get_default_model_name(self) -> str:
        """Get the default model name."""
        if not self.loaded_models:
            raise HTTPException(status_code=500, detail="No models loaded")
        return list(self.loaded_models.keys())[0]
    
    def _prepare_features(self, features: Dict[str, Any], model_name: str) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            metadata = self.model_metadata[model_name]
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
        
        logger.info(f"Starting Auto ML Model API on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app 