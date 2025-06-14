"""
Model Registry API
FastAPI endpoints for model registry operations including metadata, lineage, and governance.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import json

from auto_ml.models.registry import ModelRegistry, ModelMetadata, ModelStatus, ModelStage
from auto_ml.core.user_management import UserManager, User, Project

logger = logging.getLogger(__name__)

# Security scheme for API authentication
security = HTTPBearer()

class ModelRegistrationRequest(BaseModel):
    """Request model for model registration."""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model (classification, regression, etc.)")
    model_path: str = Field(..., description="Path to the model file")
    feature_names: List[str] = Field(..., description="List of feature names")
    target_column: str = Field(..., description="Target column name")
    training_metrics: Dict[str, float] = Field(..., description="Training performance metrics")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    description: str = Field("", description="Model description")
    framework: str = Field("scikit-learn", description="ML framework used")
    algorithm: str = Field("unknown", description="Algorithm name")
    parent_model_id: Optional[str] = Field(None, description="Parent model ID for lineage")
    tags: Optional[List[str]] = Field(None, description="Model tags")
    project_id: Optional[str] = Field(None, description="Project ID")

class ModelStatusUpdateRequest(BaseModel):
    """Request model for status updates."""
    status: ModelStatus = Field(..., description="New model status")
    notes: str = Field("", description="Optional notes about the change")

class ModelPromotionRequest(BaseModel):
    """Request model for model promotion."""
    target_stage: ModelStage = Field(..., description="Target stage")
    notes: str = Field("", description="Optional notes about the promotion")

class PerformanceRecordRequest(BaseModel):
    """Request model for performance recording."""
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    data_version: str = Field(..., description="Version of data used for evaluation")
    environment: str = Field(..., description="Environment where metrics were collected")
    drift_score: Optional[float] = Field(None, description="Data drift score")

class ModelRegistryAPI:
    """
    FastAPI-based REST API for model registry operations.
    
    This class provides:
    - Model registration and metadata management
    - Model status and stage management
    - Performance tracking and drift detection
    - Model lineage and search capabilities
    - Governance and compliance tracking
    - Audit logging for all operations
    
    Attributes:
        app (FastAPI): FastAPI application instance
        registry (ModelRegistry): Model registry instance
        user_manager (UserManager): User management system
    """
    
    def __init__(self, registry_path: str = "models/registry", host: str = "0.0.0.0", port: int = 8001):
        """
        Initialize the Model Registry API.
        
        Args:
            registry_path (str): Path for registry database
            host (str): Host address for the API server
            port (int): Port number for the API server
        """
        self.registry_path = registry_path
        self.host = host
        self.port = port
        
        # Initialize components
        self.registry = ModelRegistry(registry_path)
        self.user_manager = UserManager()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Auto ML Model Registry API",
            description="Enterprise-grade model registry with metadata tracking, lineage, and governance",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Register routes
        self._register_routes()
        
        logger.info("Model Registry API initialized successfully")
    
    async def _get_user_context(self, 
                               credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
        """
        Get user context from authentication token.
        
        Args:
            credentials: HTTP authorization credentials
            
        Returns:
            User: Authenticated user
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Simple token validation (username:token format)
            token = credentials.credentials
            
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
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    def _validate_token(self, username: str, token: str) -> bool:
        """Validate user token (simplified for development)."""
        return len(token) > 0
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Auto ML Model Registry API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=Dict[str, Any])
        async def health_check():
            """Health check endpoint."""
            try:
                # Get basic registry stats
                all_models = self.registry.list_models(limit=1000)
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "total_models": len(all_models),
                    "models_by_status": self._count_models_by_status(all_models),
                    "models_by_stage": self._count_models_by_stage(all_models),
                    "registry_path": str(self.registry.registry_path)
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/register", response_model=Dict[str, str])
        async def register_model(request: ModelRegistrationRequest,
                               user: User = Depends(self._get_user_context)):
            """Register a new model in the registry."""
            try:
                # Use project_id from request or user's default project
                project_id = request.project_id or user.default_project_id
                
                if not project_id:
                    raise HTTPException(status_code=400, detail="Project ID required")
                
                # Register the model
                model_id = self.registry.register_model(
                    model_name=request.model_name,
                    model_type=request.model_type,
                    owner=user.username,
                    project_id=project_id,
                    model_path=request.model_path,
                    feature_names=request.feature_names,
                    target_column=request.target_column,
                    training_metrics=request.training_metrics,
                    hyperparameters=request.hyperparameters,
                    description=request.description,
                    framework=request.framework,
                    algorithm=request.algorithm,
                    parent_model_id=request.parent_model_id,
                    tags=request.tags
                )
                
                return {
                    "model_id": model_id,
                    "message": f"Model {request.model_name} registered successfully",
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"Model registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", response_model=List[Dict[str, Any]])
        async def list_models(
            owner: Optional[str] = Query(None, description="Filter by owner"),
            project_id: Optional[str] = Query(None, description="Filter by project"),
            status: Optional[str] = Query(None, description="Filter by status"),
            stage: Optional[str] = Query(None, description="Filter by stage"),
            tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
            limit: int = Query(100, description="Maximum number of results"),
            user: User = Depends(self._get_user_context)
        ):
            """List models with filtering options."""
            try:
                # Parse filters
                status_enum = ModelStatus(status) if status else None
                stage_enum = ModelStage(stage) if stage else None
                tags_list = tags.split(',') if tags else None
                
                # Get models
                models = self.registry.list_models(
                    owner=owner,
                    project_id=project_id,
                    status=status_enum,
                    stage=stage_enum,
                    tags=tags_list,
                    limit=limit
                )
                
                # Convert to dict format for JSON serialization
                return [self._model_to_dict(model) for model in models]
                
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}", response_model=Dict[str, Any])
        async def get_model(model_id: str,
                          user: User = Depends(self._get_user_context)):
            """Get detailed information about a specific model."""
            try:
                model = self.registry.get_model(model_id)
                
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Check access permissions (simplified)
                if model.owner != user.username and user.role.value != "admin":
                    raise HTTPException(status_code=403, detail="Access denied")
                
                return self._model_to_dict(model)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get model {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/models/{model_id}/status", response_model=Dict[str, str])
        async def update_model_status(
            model_id: str,
            request: ModelStatusUpdateRequest,
            user: User = Depends(self._get_user_context)
        ):
            """Update model status."""
            try:
                # Check if model exists and user has access
                model = self.registry.get_model(model_id)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                if model.owner != user.username and user.role.value != "admin":
                    raise HTTPException(status_code=403, detail="Access denied")
                
                # Update status
                success = self.registry.update_model_status(
                    model_id=model_id,
                    status=request.status,
                    updated_by=user.username,
                    notes=request.notes
                )
                
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to update status")
                
                return {
                    "message": f"Model status updated to {request.status.value}",
                    "status": "success"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update model status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/models/{model_id}/promote", response_model=Dict[str, str])
        async def promote_model(
            model_id: str,
            request: ModelPromotionRequest,
            user: User = Depends(self._get_user_context)
        ):
            """Promote model to a new stage."""
            try:
                # Check if model exists and user has access
                model = self.registry.get_model(model_id)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                if model.owner != user.username and user.role.value != "admin":
                    raise HTTPException(status_code=403, detail="Access denied")
                
                # Promote model
                success = self.registry.promote_model(
                    model_id=model_id,
                    target_stage=request.target_stage,
                    approved_by=user.username,
                    notes=request.notes
                )
                
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to promote model")
                
                return {
                    "message": f"Model promoted to {request.target_stage.value}",
                    "status": "success"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to promote model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_id}/performance", response_model=Dict[str, str])
        async def record_performance(
            model_id: str,
            request: PerformanceRecordRequest,
            user: User = Depends(self._get_user_context)
        ):
            """Record model performance metrics."""
            try:
                # Check if model exists
                model = self.registry.get_model(model_id)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Record performance
                success = self.registry.record_performance(
                    model_id=model_id,
                    metrics=request.metrics,
                    data_version=request.data_version,
                    environment=request.environment,
                    drift_score=request.drift_score
                )
                
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to record performance")
                
                return {
                    "message": "Performance metrics recorded successfully",
                    "status": "success"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to record performance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}/lineage", response_model=Dict[str, Any])
        async def get_model_lineage(
            model_id: str,
            user: User = Depends(self._get_user_context)
        ):
            """Get model lineage information."""
            try:
                # Check if model exists and user has access
                model = self.registry.get_model(model_id)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                if model.owner != user.username and user.role.value != "admin":
                    raise HTTPException(status_code=403, detail="Access denied")
                
                # Get lineage
                lineage = self.registry.get_model_lineage(model_id)
                
                if not lineage:
                    return {"message": "No lineage information found"}
                
                return {
                    "model_id": lineage.model_id,
                    "parent_model_id": lineage.parent_model_id,
                    "child_model_ids": lineage.child_model_ids,
                    "training_data_version": lineage.training_data_version,
                    "code_version": lineage.code_version,
                    "experiment_id": lineage.experiment_id,
                    "changes_description": lineage.changes_description
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get model lineage: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/search", response_model=List[Dict[str, Any]])
        async def search_models(
            q: str = Query(..., description="Search query"),
            limit: int = Query(50, description="Maximum number of results"),
            user: User = Depends(self._get_user_context)
        ):
            """Search models by name, description, or tags."""
            try:
                models = self.registry.search_models(query=q, limit=limit)
                
                # Filter by user access (simplified)
                accessible_models = [
                    model for model in models 
                    if model.owner == user.username or user.role.value == "admin"
                ]
                
                return [self._model_to_dict(model) for model in accessible_models]
                
            except Exception as e:
                logger.error(f"Failed to search models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/registry/export", response_model=Dict[str, str])
        async def export_registry(
            user: User = Depends(self._get_user_context)
        ):
            """Export the entire registry to JSON."""
            try:
                # Only allow admin users to export
                if user.role.value != "admin":
                    raise HTTPException(status_code=403, detail="Admin access required")
                
                export_path = f"models/registry/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                success = self.registry.export_registry(export_path)
                
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to export registry")
                
                return {
                    "message": "Registry exported successfully",
                    "export_path": export_path,
                    "status": "success"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to export registry: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _model_to_dict(self, model: ModelMetadata) -> Dict[str, Any]:
        """Convert ModelMetadata to dictionary for JSON serialization."""
        return {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "version": model.version,
            "model_type": model.model_type,
            "description": model.description,
            "owner": model.owner,
            "project_id": model.project_id,
            "team": model.owner,
            "framework": model.framework,
            "algorithm": model.algorithm,
            "hyperparameters": model.hyperparameters,
            "feature_names": model.feature_names,
            "target_column": model.target_column,
            "training_metrics": model.training_metrics,
            "validation_metrics": model.validation_metrics,
            "test_metrics": model.test_metrics,
            "model_path": model.model_path,
            "feature_pipeline_path": model.feature_pipeline_path,
            "preprocessing_config": model.preprocessing_config,
            "model_size_mb": model.model_size_mb,
            "model_hash": model.model_hash,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "status": model.status.value,
            "stage": model.stage.value,
            "tags": model.tags,
            "parent_model_id": model.parent_model_id,
            "training_data_version": model.training_data_version,
            "code_version": model.code_version,
            "approved_by": model.approved_by,
            "approved_at": model.approved_at,
            "compliance_tags": model.compliance_tags
        }
    
    def _count_models_by_status(self, models: List[ModelMetadata]) -> Dict[str, int]:
        """Count models by status."""
        counts = {}
        for model in models:
            status = model.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts
    
    def _count_models_by_stage(self, models: List[ModelMetadata]) -> Dict[str, int]:
        """Count models by stage."""
        counts = {}
        for model in models:
            stage = model.stage.value
            counts[stage] = counts.get(stage, 0) + 1
        return counts
    
    def run(self):
        """Run the API server."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port) 