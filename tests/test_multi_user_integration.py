"""
Test Multi-User Integration
Comprehensive tests for user management, project isolation, and API integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from auto_ml.core.pipeline import AutoMLPipeline
from auto_ml.core.user_management import UserManager, User, Project, UserRole
from auto_ml.deployment.api.model_api import ModelAPI, UserContext
from auto_ml.core.config import ConfigManager


class TestMultiUserPipeline:
    """Test multi-user pipeline functionality."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def user_manager(self, temp_project_dir):
        """Create user manager with temporary data directory."""
        data_dir = Path(temp_project_dir) / "users"
        return UserManager(str(data_dir))
    
    @pytest.fixture
    def test_user(self, user_manager):
        """Create test user."""
        user_manager.create_user("testuser", "test@example.com", UserRole.USER)
        return user_manager.get_user("testuser")
    
    @pytest.fixture
    def test_project(self, user_manager, test_user):
        """Create test project."""
        project_id = user_manager.create_project(
            "Test Project", 
            test_user.username, 
            "Test project for integration testing"
        )
        return user_manager.get_project(project_id)
    
    def test_pipeline_user_context_resolution(self, user_manager, test_user):
        """Test pipeline user context resolution."""
        # Test with user object
        pipeline = AutoMLPipeline(user_context=test_user)
        assert pipeline.user.username == test_user.username
        assert pipeline.user.role == test_user.role
        
        # Test with user ID string
        pipeline = AutoMLPipeline(user_context=test_user.username)
        assert pipeline.user.username == test_user.username
        
        # Test with None (should use default admin)
        pipeline = AutoMLPipeline()
        assert pipeline.user.username == "admin"
    
    def test_pipeline_project_context_resolution(self, user_manager, test_user, test_project):
        """Test pipeline project context resolution."""
        # Test with project object
        pipeline = AutoMLPipeline(user_context=test_user, project_context=test_project)
        assert pipeline.project.project_id == test_project.project_id
        
        # Test with project ID string
        pipeline = AutoMLPipeline(user_context=test_user, project_context=test_project.project_id)
        assert pipeline.project.project_id == test_project.project_id
        
        # Test with None (should create default project)
        pipeline = AutoMLPipeline(user_context=test_user)
        assert pipeline.project is not None
        assert pipeline.project.name == f"{test_user.username}_default_project"
    
    def test_pipeline_project_directory_structure(self, user_manager, test_user, test_project):
        """Test pipeline creates proper project directory structure."""
        pipeline = AutoMLPipeline(user_context=test_user, project_context=test_project)
        
        # Check project directory structure
        assert pipeline.project_dir.exists()
        assert pipeline.data_dir.exists()
        assert pipeline.models_dir.exists()
        assert pipeline.results_dir.exists()
        assert pipeline.monitoring_dir.exists()
        
        # Check directory paths
        expected_project_dir = Path(f"projects/{test_user.username}/{test_project.project_id}")
        assert pipeline.project_dir == expected_project_dir
    
    def test_pipeline_permission_checking(self, user_manager, test_user):
        """Test pipeline enforces project permissions."""
        # Create project owned by different user
        other_user = user_manager.create_user("otheruser", "other@example.com", UserRole.USER)
        other_project_id = user_manager.create_project("Other Project", other_user.username, "Other user's project")
        
        # Should fail when test user tries to access other user's project
        with pytest.raises(ValueError, match="does not have access to project"):
            AutoMLPipeline(user_context=test_user, project_context=other_project_id)
    
    @patch('auto_ml.core.pipeline.AdultIncomeDataIngestion')
    @patch('auto_ml.core.pipeline.StandardFeatureEngineering')
    @patch('auto_ml.core.pipeline.ClassificationModelTraining')
    @patch('auto_ml.core.pipeline.DriftDetection')
    def test_pipeline_run_with_mock_components(self, mock_drift, mock_training, mock_fe, mock_ingestion,
                                             user_manager, test_user, test_project):
        """Test pipeline run with mocked components."""
        # Mock data ingestion
        mock_ingestion_instance = MagicMock()
        mock_ingestion_instance.load_data.return_value = (
            pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]}),
            pd.DataFrame({'feature1': [4, 5], 'target': [1, 0]})
        )
        mock_ingestion_instance.get_categorical_columns.return_value = []
        mock_ingestion_instance.get_numerical_columns.return_value = ['feature1']
        mock_ingestion_instance.get_target_column.return_value = 'target'
        mock_ingestion.return_value = mock_ingestion_instance
        
        # Mock feature engineering
        mock_fe_instance = MagicMock()
        mock_fe_instance.fit_transform.return_value = pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]})
        mock_fe_instance.transform.return_value = pd.DataFrame({'feature1': [4, 5], 'target': [1, 0]})
        mock_fe_instance.get_feature_names.return_value = ['feature1']
        mock_fe.return_value = mock_fe_instance
        
        # Mock model training
        mock_training_instance = MagicMock()
        mock_training_instance.train_models.return_value = {'lightgbm': {'auc': 0.85}}
        mock_training_instance.best_model_name = 'lightgbm'
        mock_training_instance.best_model.predict.return_value = np.array([1, 0])
        mock_training.return_value = mock_training_instance
        
        # Mock drift detection
        mock_drift_instance = MagicMock()
        mock_drift_instance.detect_drift.return_value = {
            'drift_detected': False,
            'overall_drift_score': 0.02,
            'alerts': []
        }
        mock_drift.return_value = mock_drift_instance
        
        # Run pipeline
        pipeline = AutoMLPipeline(user_context=test_user, project_context=test_project)
        results = pipeline.run()
        
        # Verify results structure
        assert 'user' in results
        assert 'project' in results
        assert 'data' in results
        assert 'model' in results
        assert 'results' in results
        assert 'monitoring' in results
        assert 'paths' in results
        
        # Verify user and project info
        assert results['user']['username'] == test_user.username
        assert results['project']['project_id'] == test_project.project_id
        
        # Verify project was updated
        updated_project = user_manager.get_project(test_project.project_id)
        assert len(updated_project.experiments) > 0
    
    def test_pipeline_project_summary(self, user_manager, test_user, test_project):
        """Test pipeline project summary functionality."""
        pipeline = AutoMLPipeline(user_context=test_user, project_context=test_project)
        summary = pipeline.get_project_summary()
        
        assert summary['user']['username'] == test_user.username
        assert summary['project']['project_id'] == test_project.project_id
        assert 'paths' in summary


class TestMultiUserAPI:
    """Test multi-user API functionality."""
    
    @pytest.fixture
    def temp_api_dir(self):
        """Create temporary directory for API testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def api(self, temp_api_dir):
        """Create API instance with temporary directory."""
        return ModelAPI(models_base_dir=temp_api_dir)
    
    @pytest.fixture
    def test_user(self, api):
        """Create test user for API."""
        api.user_manager.create_user("apiuser", "api@example.com", UserRole.USER)
        return api.user_manager.get_user("apiuser")
    
    @pytest.fixture
    def test_project(self, api, test_user):
        """Create test project for API."""
        project_id = api.user_manager.create_project(
            "API Test Project", 
            test_user.username, 
            "Test project for API testing"
        )
        return api.user_manager.get_project(project_id)
    
    def test_api_user_context_validation(self, api, test_user):
        """Test API user context validation."""
        # Test valid token
        valid_token = f"{test_user.username}:valid_token"
        with patch.object(api, '_validate_token', return_value=True):
            context = api._get_user_context(MagicMock(credentials=valid_token))
            assert context.user.username == test_user.username
        
        # Test invalid token format
        invalid_token = "invalid_token_format"
        with pytest.raises(Exception):
            api._get_user_context(MagicMock(credentials=invalid_token))
        
        # Test non-existent user
        non_existent_token = "nonexistent:token"
        with patch.object(api, '_validate_token', return_value=True):
            with pytest.raises(Exception):
                api._get_user_context(MagicMock(credentials=non_existent_token))
    
    def test_api_project_access_control(self, api, test_user, test_project):
        """Test API project access control."""
        # Test valid access
        valid_token = f"{test_user.username}:valid_token"
        with patch.object(api, '_validate_token', return_value=True):
            context = api._get_user_context(
                MagicMock(credentials=valid_token), 
                project_id=test_project.project_id
            )
            assert context.project.project_id == test_project.project_id
        
        # Test access to non-existent project
        with patch.object(api, '_validate_token', return_value=True):
            with pytest.raises(Exception):
                api._get_user_context(
                    MagicMock(credentials=valid_token), 
                    project_id="nonexistent_project"
                )
    
    def test_api_project_models_directory(self, api, test_user, test_project):
        """Test API project models directory structure."""
        models_dir = api._get_project_models_dir(test_user, test_project)
        expected_path = Path(api.models_base_dir) / test_user.username / test_project.project_id / "models"
        assert models_dir == expected_path
    
    def test_api_model_persistence_isolation(self, api, test_user, test_project):
        """Test API model persistence isolation between projects."""
        # Create second user and project
        other_user = api.user_manager.create_user("otheruser", "other@example.com", UserRole.USER)
        other_project_id = api.user_manager.create_project("Other Project", other_user.username, "Other project")
        other_project = api.user_manager.get_project(other_project_id)
        
        # Get model persistence for both projects
        persistence1 = api._get_model_persistence(test_user, test_project)
        persistence2 = api._get_model_persistence(other_user, other_project)
        
        # Verify different directories
        assert persistence1.models_dir != persistence2.models_dir
    
    def test_api_health_check(self, api):
        """Test API health check endpoint."""
        # Mock the health check endpoint
        with api.app.test_client() as client:
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert 'status' in data
            assert 'users_online' in data
            assert 'projects_active' in data
            assert data['status'] == 'healthy'
    
    def test_api_authentication_required(self, api):
        """Test API endpoints require authentication."""
        # Test prediction endpoint without auth
        with api.app.test_client() as client:
            response = client.post("/predict", json={
                "features": {"feature1": 1.0},
                "project_id": "test_project"
            })
            assert response.status_code == 401  # Unauthorized
    
    def test_api_token_validation(self, api):
        """Test API token validation."""
        # Test valid token
        assert api._validate_token("testuser", "valid_token") == True
        
        # Test empty token
        assert api._validate_token("testuser", "") == False


class TestMultiUserIntegration:
    """Test full multi-user integration."""
    
    @pytest.fixture
    def temp_integration_dir(self):
        """Create temporary directory for integration testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_pipeline_to_api_integration(self, temp_integration_dir):
        """Test integration between pipeline and API."""
        # Create user manager
        user_manager = UserManager(str(Path(temp_integration_dir) / "users"))
        
        # Create test user and project
        user_manager.create_user("integration_user", "integration@example.com", UserRole.USER)
        user = user_manager.get_user("integration_user")
        project_id = user_manager.create_project("Integration Project", user.username, "Integration test")
        project = user_manager.get_project(project_id)
        
        # Create API with same user manager
        api = ModelAPI(models_base_dir=temp_integration_dir)
        
        # Verify both use same user data
        api_user = api.user_manager.get_user("integration_user")
        assert api_user.username == user.username
        assert api_user.role == user.role
        
        api_project = api.user_manager.get_project(project_id)
        assert api_project.project_id == project.project_id
    
    def test_project_isolation(self, temp_integration_dir):
        """Test complete project isolation."""
        # Create two users with their own projects
        user_manager = UserManager(str(Path(temp_integration_dir) / "users"))
        
        # User 1
        user_manager.create_user("user1", "user1@example.com", UserRole.USER)
        user1 = user_manager.get_user("user1")
        project1_id = user_manager.create_project("Project 1", user1.username, "User 1 project")
        project1 = user_manager.get_project(project1_id)
        
        # User 2
        user_manager.create_user("user2", "user2@example.com", UserRole.USER)
        user2 = user_manager.get_user("user2")
        project2_id = user_manager.create_project("Project 2", user2.username, "User 2 project")
        project2 = user_manager.get_project(project2_id)
        
        # Create pipelines for both users
        pipeline1 = AutoMLPipeline(user_context=user1, project_context=project1)
        pipeline2 = AutoMLPipeline(user_context=user2, project_context=project2)
        
        # Verify different project directories
        assert pipeline1.project_dir != pipeline2.project_dir
        assert pipeline1.user.username != pipeline2.user.username
        assert pipeline1.project.project_id != pipeline2.project.project_id
        
        # Verify user manager enforces isolation
        assert user_manager.check_permission(user1.username, project1_id, "read") == True
        assert user_manager.check_permission(user1.username, project2_id, "read") == False
        assert user_manager.check_permission(user2.username, project2_id, "read") == True
        assert user_manager.check_permission(user2.username, project1_id, "read") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 