"""
Model Registry System
Enterprise-grade model registry with metadata tracking, lineage, and governance.
"""

import os
import json
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import yaml

from auto_ml.core.exceptions import ModelRegistryError

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model deployment status."""
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ModelStage(Enum):
    """Model lifecycle stage."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    # Basic Information
    model_id: str
    model_name: str
    version: str
    model_type: str
    description: str
    
    # Ownership and Access
    owner: str
    project_id: str
    
    # Technical Details
    framework: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    target_column: str
    
    # Performance Metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Model Artifacts
    model_path: str
    model_size_mb: float
    model_hash: str
    created_at: str
    updated_at: str
    
    # Optional fields with defaults
    team: Optional[str] = None
    feature_pipeline_path: Optional[str] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    status: ModelStatus = ModelStatus.DRAFT
    stage: ModelStage = ModelStage.DEVELOPMENT
    tags: List[str] = None
    parent_model_id: Optional[str] = None
    training_data_version: Optional[str] = None
    code_version: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    compliance_tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.compliance_tags is None:
            self.compliance_tags = []

@dataclass
class ModelLineage:
    """Model lineage information."""
    model_id: str
    parent_model_id: Optional[str]
    child_model_ids: List[str]
    training_data_version: str
    code_version: str
    experiment_id: Optional[str] = None
    changes_description: str = ""

@dataclass
class ModelPerformance:
    """Model performance tracking."""
    model_id: str
    timestamp: str
    metrics: Dict[str, float]
    data_version: str
    environment: str
    drift_score: Optional[float] = None

class ModelRegistry:
    """
    Enterprise-grade model registry with comprehensive metadata tracking.
    
    This class provides:
    - Model metadata management with full lineage tracking
    - Performance monitoring and drift detection
    - Model lifecycle management (draft -> production)
    - Governance and compliance tracking
    - Advanced search and filtering capabilities
    - Audit logging for all operations
    
    Attributes:
        registry_path (Path): Path to the registry database
        db_connection (sqlite3.Connection): Database connection
        models_dir (Path): Directory for storing model files
    """
    
    def __init__(self, registry_path: str = "models/registry", models_dir: str = "models"):
        """
        Initialize the model registry.
        
        Args:
            registry_path (str): Path for registry database and metadata
            models_dir (str): Directory for storing model files
        """
        self.registry_path = Path(registry_path)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.registry_path / "model_registry.db"
        self._init_database()
        
        logger.info(f"Model registry initialized at {self.registry_path}")
    
    def register_model(self, 
                      model_name: str,
                      model_type: str,
                      owner: str,
                      project_id: str,
                      model_path: str,
                      feature_names: List[str],
                      target_column: str,
                      training_metrics: Dict[str, float],
                      hyperparameters: Dict[str, Any],
                      description: str = "",
                      framework: str = "scikit-learn",
                      algorithm: str = "unknown",
                      parent_model_id: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model (classification, regression, etc.)
            owner (str): Model owner/creator
            project_id (str): Project identifier
            model_path (str): Path to the model file
            feature_names (List[str]): List of feature names
            target_column (str): Target column name
            training_metrics (Dict[str, float]): Training performance metrics
            hyperparameters (Dict[str, Any]): Model hyperparameters
            description (str): Model description
            framework (str): ML framework used
            algorithm (str): Algorithm name
            parent_model_id (Optional[str]): Parent model ID for lineage
            tags (Optional[List[str]]): Model tags
            **kwargs: Additional metadata fields
            
        Returns:
            str: Model ID
            
        Raises:
            ModelRegistryError: If registration fails
        """
        try:
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Validate model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                raise ModelRegistryError(f"Model file not found: {model_path}")
            
            # Calculate model hash and size
            model_hash = self._calculate_file_hash(model_file)
            model_size_mb = model_file.stat().st_size / (1024 * 1024)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=self._generate_version(model_name),
                model_type=model_type,
                description=description,
                owner=owner,
                project_id=project_id,
                framework=framework,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                feature_names=feature_names,
                target_column=target_column,
                training_metrics=training_metrics,
                validation_metrics=kwargs.get('validation_metrics', {}),
                test_metrics=kwargs.get('test_metrics', {}),
                model_path=model_path,
                model_size_mb=model_size_mb,
                model_hash=model_hash,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                status=ModelStatus.TRAINED,
                stage=ModelStage.DEVELOPMENT,
                feature_pipeline_path=kwargs.get('feature_pipeline_path'),
                preprocessing_config=kwargs.get('preprocessing_config'),
                tags=tags or [],
                parent_model_id=parent_model_id,
                training_data_version=kwargs.get('training_data_version'),
                code_version=kwargs.get('code_version'),
                approved_by=kwargs.get('approved_by'),
                approved_at=kwargs.get('approved_at'),
                compliance_tags=kwargs.get('compliance_tags', [])
            )
            
            # Store in database
            self._store_model_metadata(metadata)
            
            # Create lineage entry
            if parent_model_id:
                self._create_lineage_entry(metadata, parent_model_id)
            
            # Log audit event
            self._log_audit_event("model_registered", model_id, owner, metadata)
            
            logger.info(f"Model {model_name} registered with ID {model_id}")
            return model_id
            
        except Exception as e:
            raise ModelRegistryError(f"Failed to register model {model_name}: {e}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            Optional[ModelMetadata]: Model metadata or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT metadata FROM models WHERE model_id = ?",
                    (model_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    metadata_dict = json.loads(result[0])
                    # Convert status and stage back to enums if present
                    if 'status' in metadata_dict and isinstance(metadata_dict['status'], str):
                        metadata_dict['status'] = ModelStatus(metadata_dict['status'])
                    if 'stage' in metadata_dict and isinstance(metadata_dict['stage'], str):
                        metadata_dict['stage'] = ModelStage(metadata_dict['stage'])
                    return ModelMetadata(**metadata_dict)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    def list_models(self, 
                   owner: Optional[str] = None,
                   project_id: Optional[str] = None,
                   status: Optional[ModelStatus] = None,
                   stage: Optional[ModelStage] = None,
                   tags: Optional[List[str]] = None,
                   limit: int = 100) -> List[ModelMetadata]:
        """
        List models with filtering options.
        
        Args:
            owner (Optional[str]): Filter by owner
            project_id (Optional[str]): Filter by project
            status (Optional[ModelStatus]): Filter by status
            stage (Optional[ModelStage]): Filter by stage
            tags (Optional[List[str]]): Filter by tags
            limit (int): Maximum number of results
            
        Returns:
            List[ModelMetadata]: List of matching models
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = "SELECT metadata FROM models WHERE 1=1"
                params = []
                
                if owner:
                    query += " AND json_extract(metadata, '$.owner') = ?"
                    params.append(owner)
                
                if project_id:
                    query += " AND json_extract(metadata, '$.project_id') = ?"
                    params.append(project_id)
                
                if status:
                    query += " AND json_extract(metadata, '$.status') = ?"
                    params.append(status.value)
                
                if stage:
                    query += " AND json_extract(metadata, '$.stage') = ?"
                    params.append(stage.value)
                
                if tags:
                    for tag in tags:
                        query += " AND json_extract(metadata, '$.tags') LIKE ?"
                        params.append(f'%"{tag}"%')
                
                query += " ORDER BY json_extract(metadata, '$.created_at') DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                models = []
                for result in results:
                    metadata_dict = json.loads(result[0])
                    models.append(ModelMetadata(**metadata_dict))
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def update_model_status(self, model_id: str, status: ModelStatus, 
                           updated_by: str, notes: str = "") -> bool:
        """
        Update model status.
        
        Args:
            model_id (str): Model ID
            status (ModelStatus): New status
            updated_by (str): User updating the status
            notes (str): Optional notes about the change
            
        Returns:
            bool: True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current metadata
                cursor.execute(
                    "SELECT metadata FROM models WHERE model_id = ?",
                    (model_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    raise ModelRegistryError(f"Model {model_id} not found")
                
                metadata_dict = json.loads(result[0])
                metadata = ModelMetadata(**metadata_dict)
                
                # Update status
                metadata.status = status
                metadata.updated_at = datetime.now().isoformat()
                
                # Convert dataclass to dict and enums to values
                meta_dict = asdict(metadata)
                if isinstance(meta_dict.get('status'), Enum):
                    meta_dict['status'] = meta_dict['status'].value
                if isinstance(meta_dict.get('stage'), Enum):
                    meta_dict['stage'] = meta_dict['stage'].value
                
                # Update in database
                cursor.execute(
                    "UPDATE models SET metadata = ?, updated_at = ? WHERE model_id = ?",
                    (self._serialize_metadata(metadata), datetime.now().isoformat(), model_id)
                )
                
                conn.commit()
                
                # Log audit event
                self._log_audit_event("status_updated", model_id, updated_by, {
                    "old_status": metadata_dict["status"],
                    "new_status": status.value,
                    "notes": notes
                })
                
                logger.info(f"Model {model_id} status updated to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update model status: {e}")
            return False
    
    def promote_model(self, model_id: str, target_stage: ModelStage, 
                     approved_by: str, notes: str = "") -> bool:
        """
        Promote model to a new stage (e.g., development -> staging -> production).
        
        Args:
            model_id (str): Model ID
            target_stage (ModelStage): Target stage
            approved_by (str): User approving the promotion
            notes (str): Optional notes about the promotion
            
        Returns:
            bool: True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current metadata
                cursor.execute(
                    "SELECT metadata FROM models WHERE model_id = ?",
                    (model_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    raise ModelRegistryError(f"Model {model_id} not found")
                
                metadata_dict = json.loads(result[0])
                metadata = ModelMetadata(**metadata_dict)
                
                # Update stage and approval info
                metadata.stage = target_stage
                metadata.approved_by = approved_by
                metadata.approved_at = datetime.now().isoformat()
                metadata.updated_at = datetime.now().isoformat()
                
                # Update status based on stage
                if target_stage == ModelStage.PRODUCTION:
                    metadata.status = ModelStatus.DEPLOYED
                
                # Update in database
                cursor.execute(
                    "UPDATE models SET metadata = ?, updated_at = ? WHERE model_id = ?",
                    (self._serialize_metadata(metadata), datetime.now().isoformat(), model_id)
                )
                
                conn.commit()
                
                # Log audit event
                self._log_audit_event("model_promoted", model_id, approved_by, {
                    "old_stage": metadata_dict["stage"],
                    "new_stage": target_stage.value,
                    "notes": notes
                })
                
                logger.info(f"Model {model_id} promoted to {target_stage.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def record_performance(self, model_id: str, metrics: Dict[str, float],
                          data_version: str, environment: str,
                          drift_score: Optional[float] = None) -> bool:
        """
        Record model performance metrics.
        
        Args:
            model_id (str): Model ID
            metrics (Dict[str, float]): Performance metrics
            data_version (str): Version of data used for evaluation
            environment (str): Environment where metrics were collected
            drift_score (Optional[float]): Data drift score
            
        Returns:
            bool: True if successful
        """
        try:
            performance = ModelPerformance(
                model_id=model_id,
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
                data_version=data_version,
                environment=environment,
                drift_score=drift_score
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO model_performance 
                       (model_id, timestamp, metrics, data_version, environment, drift_score)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (model_id, performance.timestamp, json.dumps(metrics),
                     data_version, environment, drift_score)
                )
                conn.commit()
            
            logger.info(f"Performance recorded for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record performance for model {model_id}: {e}")
            return False
    
    def get_model_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """
        Get model lineage information.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            Optional[ModelLineage]: Lineage information or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM model_lineage WHERE model_id = ?",
                    (model_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    return ModelLineage(
                        model_id=result[0],
                        parent_model_id=result[1],
                        child_model_ids=json.loads(result[2]) if result[2] else [],
                        training_data_version=result[3],
                        code_version=result[4],
                        experiment_id=result[5],
                        changes_description=result[6] or ""
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get lineage for model {model_id}: {e}")
            return None
    
    def search_models(self, query: str, limit: int = 50) -> List[ModelMetadata]:
        """
        Search models by name, description, or tags.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[ModelMetadata]: Matching models
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Search in name, description, and tags
                search_query = f"%{query}%"
                cursor.execute(
                    """SELECT metadata FROM models 
                       WHERE json_extract(metadata, '$.model_name') LIKE ?
                          OR json_extract(metadata, '$.description') LIKE ?
                          OR json_extract(metadata, '$.tags') LIKE ?
                       ORDER BY json_extract(metadata, '$.created_at') DESC 
                       LIMIT ?""",
                    (search_query, search_query, search_query, limit)
                )
                
                results = cursor.fetchall()
                models = []
                for result in results:
                    metadata_dict = json.loads(result[0])
                    models.append(ModelMetadata(**metadata_dict))
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []
    
    def export_registry(self, export_path: str) -> bool:
        """
        Export the entire registry to a JSON file.
        
        Args:
            export_path (str): Path to export file
            
        Returns:
            bool: True if successful
        """
        try:
            models = self.list_models(limit=10000)  # Get all models
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_models": len(models),
                "models": [asdict(model) for model in models],
                "registry_version": "1.0"
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Registry exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            return False
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT PRIMARY KEY,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Model lineage table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_lineage (
                        model_id TEXT PRIMARY KEY,
                        parent_model_id TEXT,
                        child_model_ids TEXT,
                        training_data_version TEXT,
                        code_version TEXT,
                        experiment_id TEXT,
                        changes_description TEXT,
                        FOREIGN KEY (model_id) REFERENCES models (model_id)
                    )
                """)
                
                # Model performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        data_version TEXT NOT NULL,
                        environment TEXT NOT NULL,
                        drift_score REAL,
                        FOREIGN KEY (model_id) REFERENCES models (model_id)
                    )
                """)
                
                # Audit log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        model_id TEXT,
                        user_id TEXT NOT NULL,
                        details TEXT,
                        FOREIGN KEY (model_id) REFERENCES models (model_id)
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_owner ON models(json_extract(metadata, '$.owner'))")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_project ON models(json_extract(metadata, '$.project_id'))")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models(json_extract(metadata, '$.status'))")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_stage ON models(json_extract(metadata, '$.stage'))")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_model ON model_performance(model_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON model_performance(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_log(model_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
                
                conn.commit()
                
        except Exception as e:
            raise ModelRegistryError(f"Failed to initialize database: {e}")
    
    def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO models (model_id, metadata, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (metadata.model_id, self._serialize_metadata(metadata), 
                 metadata.created_at, metadata.updated_at)
            )
            conn.commit()
    
    def _create_lineage_entry(self, metadata: ModelMetadata, parent_model_id: str):
        """Create lineage entry for a model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create lineage entry
            cursor.execute(
                """INSERT INTO model_lineage 
                   (model_id, parent_model_id, training_data_version, code_version)
                   VALUES (?, ?, ?, ?)""",
                (metadata.model_id, parent_model_id, 
                 metadata.training_data_version or "unknown",
                 metadata.code_version or "unknown")
            )
            
            # Update parent's child list
            cursor.execute(
                "SELECT child_model_ids FROM model_lineage WHERE model_id = ?",
                (parent_model_id,)
            )
            result = cursor.fetchone()
            
            if result:
                child_ids = json.loads(result[0]) if result[0] else []
                child_ids.append(metadata.model_id)
                cursor.execute(
                    "UPDATE model_lineage SET child_model_ids = ? WHERE model_id = ?",
                    (json.dumps(child_ids), parent_model_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO model_lineage (model_id, child_model_ids) VALUES (?, ?)",
                    (parent_model_id, json.dumps([metadata.model_id]))
                )
            
            conn.commit()
    
    def _log_audit_event(self, event_type: str, model_id: str, user_id: str, details: Dict[str, Any]):
        """Log an audit event."""
        # Convert dataclass to dict if needed
        if hasattr(details, '__dataclass_fields__'):
            details = asdict(details)
        # Convert enums to their values in details
        for k, v in details.items():
            if isinstance(v, Enum):
                details[k] = v.value
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO audit_log (timestamp, event_type, model_id, user_id, details) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), event_type, model_id, user_id, json.dumps(details))
            )
            conn.commit()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _generate_version(self, model_name: str) -> str:
        """Generate a new version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{timestamp}"

    def _convert_enums_for_json(self, obj):
        """Convert enum values to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_enums_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums_for_json(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    def _serialize_metadata(self, metadata: ModelMetadata) -> str:
        """Serialize model metadata to JSON string with enum conversion."""
        metadata_dict = asdict(metadata)
        return json.dumps(self._convert_enums_for_json(metadata_dict)) 