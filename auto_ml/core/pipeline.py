"""
Auto ML Pipeline Orchestration with Monitoring and Multi-User Support
Production-ready pipeline orchestration integrating model monitoring, drift detection, and user/project isolation.
"""

import logging
from typing import Optional, Dict, Any, Union
import pandas as pd
from pathlib import Path
from datetime import datetime

from .user_management import UserManager, User, Project
from .config import ConfigManager, get_config
from auto_ml import (
    AdultIncomeDataIngestion,
    StandardFeatureEngineering,
    ClassificationModelTraining
)
from auto_ml.monitoring.drift_detection import DriftDetection

logger = logging.getLogger(__name__)

class AutoMLPipeline:
    """
    Orchestrates the full Auto ML pipeline with monitoring integration and multi-user support.
    
    Features:
    - User and project context isolation
    - Project-specific data and model storage
    - Role-based access control
    - Monitoring and drift detection
    - Backward compatibility for single-user mode
    """
    
    def __init__(self, 
                 user_context: Optional[Union[str, User]] = None,
                 project_context: Optional[Union[str, Project]] = None,
                 user_manager: Optional[UserManager] = None,
                 config_manager: Optional[ConfigManager] = None,
                 monitoring_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Auto ML pipeline with user and project context.
        
        Args:
            user_context: User ID string or User object. If None, uses default admin user.
            project_context: Project ID string or Project object. If None, creates/uses default project.
            user_manager: UserManager instance. If None, creates new instance.
            config_manager: Configuration manager instance. If None, uses global config.
            monitoring_config: Monitoring configuration dictionary
        """
        # Initialize configuration
        self.config_manager = config_manager or get_config()
        
        # Initialize user management
        self.user_manager = user_manager or UserManager()
        
        # Resolve user context
        self.user = self._resolve_user_context(user_context)
        
        # Resolve project context
        self.project = self._resolve_project_context(project_context)
        
        # Set up project-specific paths
        self.project_dir = Path(f"projects/{self.user.username}/{self.project.project_id}")
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Project-specific directories
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        self.monitoring_dir = self.results_dir / "monitoring"
        
        for dir_path in [self.data_dir, self.models_dir, self.results_dir, self.monitoring_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Monitoring configuration
        self.monitoring_config = monitoring_config or {
            'drift_threshold': 0.05,
            'wasserstein_threshold': 0.1,
            'chi2_threshold': 0.05,
            'performance_threshold': 0.1,
            'monitor_features': True,
            'monitor_performance': True,
            'monitor_predictions': True,
            'alert_on_drift': True
        }
        
        # Initialize monitoring with project-specific paths
        self.monitoring = DriftDetection(self.monitoring_config)
        
        logger.info(f"Pipeline initialized for user '{self.user.username}' and project '{self.project.name}' ({self.project.project_id})")
        logger.info(f"Project directory: {self.project_dir}")
    
    def _resolve_user_context(self, user_context: Optional[Union[str, User]]) -> User:
        """Resolve user context from string ID or User object."""
        if user_context is None:
            # Use default admin user
            user = self.user_manager.get_user("admin")
            if user is None:
                raise ValueError("No user context provided and no default admin user found")
            return user
        
        if isinstance(user_context, User):
            return user_context
        
        if isinstance(user_context, str):
            user = self.user_manager.get_user(user_context)
            if user is None:
                raise ValueError(f"User not found: {user_context}")
            return user
        
        raise ValueError(f"Invalid user context type: {type(user_context)}")
    
    def _resolve_project_context(self, project_context: Optional[Union[str, Project]]) -> Project:
        """Resolve project context from string ID or Project object."""
        if project_context is None:
            # Create or get default project for the user
            default_project_name = f"{self.user.username}_default_project"
            user_projects = self.user_manager.list_user_projects(self.user.username)
            
            # Check if default project exists
            for project in user_projects:
                if project.name == default_project_name:
                    return project
            
            # Create default project
            project_id = self.user_manager.create_project(
                name=default_project_name,
                owner=self.user.username,
                description="Default project for Auto ML pipeline"
            )
            if project_id is None:
                raise ValueError(f"Failed to create default project for user {self.user.username}")
            
            return self.user_manager.get_project(project_id)
        
        if isinstance(project_context, Project):
            return project_context
        
        if isinstance(project_context, str):
            project = self.user_manager.get_project(project_context)
            if project is None:
                raise ValueError(f"Project not found: {project_context}")
            
            # Check if user has access to this project
            if not self.user_manager.check_permission(self.user.username, project_context, "read"):
                raise ValueError(f"User {self.user.username} does not have access to project {project_context}")
            
            return project
        
        raise ValueError(f"Invalid project context type: {type(project_context)}")
    
    def run(self, dataset_name: str = "adult_income"):
        """
        Run the complete Auto ML pipeline for the current user and project.
        
        Args:
            dataset_name: Name of the dataset to use (default: adult_income)
            
        Returns:
            Dict containing pipeline results and artifacts
        """
        logger.info(f"Starting Auto ML pipeline for user '{self.user.username}' and project '{self.project.name}'")
        logger.info(f"Using dataset: {dataset_name}")
        
        # Update project timestamp
        self.project.updated_at = datetime.now().isoformat()
        self.user_manager.update_project(self.project.project_id, updated_at=self.project.updated_at)
        
        # 1. Data ingestion with project-specific paths
        logger.info("Step 1: Data ingestion...")
        ingestion = AdultIncomeDataIngestion({
            'data_dir': str(self.data_dir),
            'dataset_name': dataset_name
        })
        train_data, test_data = ingestion.load_data()
        logger.info(f"Data loaded: Train {train_data.shape}, Test {test_data.shape}")
        
        # Save data info to project
        data_info = {
            'dataset_name': dataset_name,
            'train_shape': train_data.shape,
            'test_shape': test_data.shape,
            'categorical_columns': ingestion.get_categorical_columns(),
            'numerical_columns': ingestion.get_numerical_columns(),
            'target_column': ingestion.get_target_column(),
            'timestamp': datetime.now().isoformat()
        }
        self._save_project_data_info(data_info)
        
        # 2. Feature engineering
        logger.info("Step 2: Feature engineering...")
        fe = StandardFeatureEngineering({
            'output_dir': str(self.results_dir),
            'feature_cache': True
        })
        cat_cols = ingestion.get_categorical_columns()
        num_cols = ingestion.get_numerical_columns()
        train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
        test_processed = fe.transform(test_data, cat_cols, num_cols)
        logger.info(f"Feature engineering completed: Train {train_processed.shape}, Test {test_processed.shape}")
        
        # 3. Model training with project-specific model storage
        logger.info("Step 3: Model training...")
        mt = ClassificationModelTraining({
            'models_dir': str(self.models_dir),
            'experiment_name': f"{self.project.project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'save_models': True
        })
        feature_names = fe.get_feature_names(cat_cols, num_cols)
        results = mt.train_models(
            train_processed, train_processed['target'],
            test_processed, test_processed['target'],
            feature_names
        )
        logger.info(f"Model training completed. Best model: {mt.best_model_name}")
        
        # Save training results to project
        self._save_training_results(results, mt.best_model_name)
        
        # 4. Monitoring: set baseline after training
        logger.info("Step 4: Setting monitoring baseline...")
        self.monitoring.set_baseline(train_processed, target_column='target')
        logger.info("Baseline set for drift detection.")
        
        # 5. Inference and drift detection
        logger.info("Step 5: Running drift detection...")
        predictions = mt.best_model.predict(test_processed[feature_names])
        drift_results = self.monitoring.detect_drift(
            test_processed, target_column='target', predictions=predictions
        )
        logger.info(f"Drift detection results: {drift_results['drift_detected']}, score: {drift_results['overall_drift_score']:.3f}")
        
        if drift_results['drift_detected']:
            logger.warning(f"Drift detected! Alerts: {drift_results['alerts']}")
        else:
            logger.info("No significant drift detected.")
        
        # 6. Save monitoring report and plot to project directory
        logger.info("Step 6: Saving monitoring reports...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.monitoring_dir / f"drift_report_{timestamp}.json"
        plot_path = self.monitoring_dir / f"drift_plot_{timestamp}.png"
        
        self.monitoring.save_drift_report(str(report_path))
        self.monitoring.plot_drift_analysis(save_path=str(plot_path))
        
        logger.info(f"Monitoring report saved: {report_path}")
        logger.info(f"Monitoring plot saved: {plot_path}")
        
        # 7. Update project with experiment results
        experiment_id = f"exp_{timestamp}"
        self.project.experiments.append(experiment_id)
        self.user_manager.update_project(
            self.project.project_id,
            experiments=self.project.experiments,
            updated_at=datetime.now().isoformat()
        )
        
        # 8. Return comprehensive results
        pipeline_results = {
            'user': {
                'username': self.user.username,
                'role': self.user.role.value
            },
            'project': {
                'project_id': self.project.project_id,
                'name': self.project.name,
                'experiment_id': experiment_id
            },
            'data': {
                'train_shape': train_processed.shape,
                'test_shape': test_processed.shape,
                'feature_count': len(feature_names),
                'dataset_name': dataset_name
            },
            'model': {
                'best_model_name': mt.best_model_name,
                'best_model_path': str(self.models_dir / f"{mt.best_model_name}.pkl"),
                'feature_names': feature_names
            },
            'results': {
                'training_results': results,
                'best_model_metrics': results.get(mt.best_model_name, {})
            },
            'monitoring': {
                'drift_detected': drift_results['drift_detected'],
                'drift_score': drift_results['overall_drift_score'],
                'alerts': drift_results.get('alerts', []),
                'report_path': str(report_path),
                'plot_path': str(plot_path)
            },
            'paths': {
                'project_dir': str(self.project_dir),
                'data_dir': str(self.data_dir),
                'models_dir': str(self.models_dir),
                'results_dir': str(self.results_dir),
                'monitoring_dir': str(self.monitoring_dir)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save pipeline results
        results_file = self.results_dir / f"pipeline_results_{timestamp}.json"
        self._save_pipeline_results(pipeline_results, str(results_file))
        
        logger.info(f"Pipeline completed successfully for user '{self.user.username}' and project '{self.project.name}'")
        logger.info(f"Results saved to: {results_file}")
        
        return pipeline_results
    
    def _save_project_data_info(self, data_info: Dict[str, Any]) -> None:
        """Save data information to project directory."""
        data_info_file = self.data_dir / "data_info.json"
        import json
        with open(data_info_file, 'w') as f:
            json.dump(data_info, f, indent=2)
    
    def _save_training_results(self, results: Dict[str, Dict[str, float]], best_model_name: str) -> None:
        """Save training results to project directory."""
        training_results = {
            'results': results,
            'best_model': best_model_name,
            'timestamp': datetime.now().isoformat(),
            'user': self.user.username,
            'project': self.project.project_id
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"training_results_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
    
    def _save_pipeline_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save complete pipeline results to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_project_summary(self) -> Dict[str, Any]:
        """Get summary of the current project."""
        return {
            'user': {
                'username': self.user.username,
                'role': self.user.role.value,
                'email': self.user.email
            },
            'project': {
                'project_id': self.project.project_id,
                'name': self.project.name,
                'description': self.project.description,
                'created_at': self.project.created_at,
                'updated_at': self.project.updated_at,
                'experiment_count': len(self.project.experiments),
                'model_count': len(self.project.models)
            },
            'paths': {
                'project_dir': str(self.project_dir),
                'data_dir': str(self.data_dir),
                'models_dir': str(self.models_dir),
                'results_dir': str(self.results_dir)
            }
        } 