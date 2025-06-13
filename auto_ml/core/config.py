"""
Configuration management for the Auto ML framework.
Handles YAML configuration files with validation and environment-specific settings.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from .exceptions import ConfigurationError

class Config:
    """
    Configuration management class for the Auto ML framework.
    
    Supports:
    - YAML configuration files
    - Environment-specific settings
    - Parameter validation
    - Default values
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path (Optional[str]): Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                raise ConfigurationError(f"Failed to load config from {self.config_path}: {e}")
        else:
            # Use default configuration
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv',
                'validation_split': 0.2,
                'random_state': 42
            },
            'features': {
                'categorical_columns': [],
                'numerical_columns': [],
                'target_column': 'target',
                'drop_columns': []
            },
            'model': {
                'task_type': 'classification',  # 'classification', 'regression'
                'algorithms': ['logistic_regression', 'random_forest', 'lightgbm'],
                'hyperparameter_optimization': True,
                'cross_validation_folds': 5,
                'random_state': 42
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'enable_hyperparameter_optimization': True
            },
            'persistence': {
                'models_dir': 'models',
                'version_format': 'v{version}_{timestamp}',
                'save_feature_pipeline': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'pipeline.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key (str): Configuration key (e.g., 'data.train_path')
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key (str): Configuration key (e.g., 'data.train_path')
            value (Any): Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path (Optional[str]): Path to save config (uses self.config_path if None)
        """
        save_path = path or self.config_path
        if not save_path:
            raise ConfigurationError("No path specified for saving configuration")
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {save_path}: {e}")
    
    def validate(self) -> None:
        """Validate configuration values."""
        required_keys = [
            'data.train_path',
            'features.target_column',
            'model.task_type',
            'model.algorithms'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ConfigurationError(f"Required configuration key missing: {key}")
        
        # Validate task type
        task_type = self.get('model.task_type')
        if task_type not in ['classification', 'regression']:
            raise ConfigurationError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
        
        # Validate algorithms
        algorithms = self.get('model.algorithms')
        if not isinstance(algorithms, list) or len(algorithms) == 0:
            raise ConfigurationError("model.algorithms must be a non-empty list")
        
        # Validate validation split
        validation_split = self.get('data.validation_split')
        if not (0 < validation_split < 1):
            raise ConfigurationError(f"validation_split must be between 0 and 1, got {validation_split}")
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            'AUTO_ML_LOG_LEVEL': 'logging.level',
            'AUTO_ML_MODELS_DIR': 'persistence.models_dir',
            'AUTO_ML_RANDOM_STATE': 'training.random_state',
            'AUTO_ML_TASK_TYPE': 'model.task_type'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self.set(config_key, env_value)
    
    def get_environment_config(self, environment: str) -> 'Config':
        """
        Get environment-specific configuration.
        
        Args:
            environment (str): Environment name (e.g., 'development', 'production')
            
        Returns:
            Config: Environment-specific configuration
        """
        env_config = Config()
        env_config.config = self.config.copy()
        
        # Load environment-specific overrides
        env_file = f"config.{environment}.yaml"
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    env_overrides = yaml.safe_load(f)
                    env_config.config.update(env_overrides)
            except Exception as e:
                raise ConfigurationError(f"Failed to load environment config {env_file}: {e}")
        
        return env_config 