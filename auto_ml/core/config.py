"""
Configuration management for the Auto ML framework.
Handles API credentials, environment settings, and framework configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration and API credentials for the Auto ML framework."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._credentials: Optional[Dict[str, Any]] = None
        self._settings: Optional[Dict[str, Any]] = None
        
    def load_credentials(self) -> Dict[str, Any]:
        """
        Load API credentials from configuration file.
        
        Returns:
            Dictionary containing API credentials
            
        Raises:
            FileNotFoundError: If credentials file doesn't exist
            yaml.YAMLError: If credentials file is malformed
        """
        if self._credentials is None:
            credentials_file = self.config_dir / "api_credentials.yaml"
            
            if not credentials_file.exists():
                logger.warning(f"Credentials file not found: {credentials_file}")
                return {}
                
            try:
                with open(credentials_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self._credentials = config.get('api_credentials', {})
                    logger.info("API credentials loaded successfully")
            except yaml.YAMLError as e:
                logger.error(f"Error parsing credentials file: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading credentials: {e}")
                raise
                
        return self._credentials
    
    def get_huggingface_token(self) -> Optional[str]:
        """Get Hugging Face API token."""
        credentials = self.load_credentials()
        return credentials.get('huggingface', {}).get('token')
    
    def get_huggingface_config(self) -> Dict[str, Any]:
        """Get complete Hugging Face configuration."""
        credentials = self.load_credentials()
        return credentials.get('huggingface', {})
    
    def get_github_config(self) -> Dict[str, Any]:
        """Get GitHub configuration."""
        credentials = self.load_credentials()
        return credentials.get('github', {})
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment platform configurations."""
        credentials = self.load_credentials()
        return credentials.get('deployment', {})
    
    def load_settings(self) -> Dict[str, Any]:
        """
        Load framework settings from configuration file.
        
        Returns:
            Dictionary containing framework settings
        """
        if self._settings is None:
            credentials_file = self.config_dir / "api_credentials.yaml"
            
            if not credentials_file.exists():
                logger.warning(f"Settings file not found: {credentials_file}")
                return {}
                
            try:
                with open(credentials_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self._settings = config.get('environments', {}).get('development', {})
                    logger.info("Framework settings loaded successfully")
            except yaml.YAMLError as e:
                logger.error(f"Error parsing settings file: {e}")
                return {}
            except Exception as e:
                logger.error(f"Error loading settings: {e}")
                return {}
                
        return self._settings
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value.
        
        Args:
            key: Setting key to retrieve
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        settings = self.load_settings()
        return settings.get(key, default)
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get_setting('debug', False)
    
    def get_log_level(self) -> str:
        """Get configured log level."""
        return self.get_setting('log_level', 'INFO')
    
    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.get_setting('cache_enabled', True)
    
    def validate_credentials(self) -> bool:
        """
        Validate that required credentials are available.
        
        Returns:
            True if all required credentials are present
        """
        credentials = self.load_credentials()
        
        # Check for required credentials
        required = ['huggingface']
        
        for service in required:
            if service not in credentials:
                logger.warning(f"Missing required service configuration: {service}")
                return False
                
        # Check for required tokens
        hf_config = credentials.get('huggingface', {})
        if not hf_config.get('token'):
            logger.warning("Missing Hugging Face API token")
            return False
            
        logger.info("All required credentials validated successfully")
        return True
    
    def get_model_cache_dir(self) -> str:
        """Get the model cache directory path."""
        hf_config = self.get_huggingface_config()
        return hf_config.get('model_cache_dir', './models/huggingface_cache')


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def get_huggingface_token() -> Optional[str]:
    """Get Hugging Face API token from global config."""
    return config_manager.get_huggingface_token()


def is_debug_mode() -> bool:
    """Check if debug mode is enabled from global config."""
    return config_manager.is_debug_mode()


def get_log_level() -> str:
    """Get log level from global config."""
    return config_manager.get_log_level() 