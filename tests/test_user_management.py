"""
Test User Management System
Comprehensive tests for multi-user support with authentication, authorization, and project isolation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

# Import the system under test
from auto_ml.core.user_management import UserManager, User, Project, UserRole

class TestUserManagement:
    """Test cases for user management system."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def user_manager(self, temp_data_dir):
        """Create user manager instance for testing."""
        return UserManager(data_dir=temp_data_dir)
    
    def test_initialization(self, user_manager):
        """Test user manager initialization."""
        assert user_manager.users is not None
        assert user_manager.projects is not None
        assert user_manager.sessions is not None
        
        # Should create default admin user
        assert "admin" in user_manager.users
        admin_user = user_manager.users["admin"]
        assert admin_user.role == UserRole.ADMIN
        assert admin_user.email == "admin@automl.local"
    
    def test_create_user(self, user_manager):
        """Test user creation."""
        # Create a new user
        success = user_manager.create_user("testuser", "test@example.com", UserRole.USER)
        assert success is True
        
        # Verify user was created
        user = user_manager.get_user("testuser")
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.is_active is True
        assert user.projects == []
    
    def test_create_duplicate_user(self, user_manager):
        """Test creating duplicate user."""
        # Create first user
        user_manager.create_user("testuser", "test@example.com")
        
        # Try to create duplicate
        success = user_manager.create_user("testuser", "test2@example.com")
        assert success is False
    
    def test_get_user(self, user_manager):
        """Test getting user by username."""
        # Create user
        user_manager.create_user("testuser", "test@example.com")
        
        # Get user
        user = user_manager.get_user("testuser")
        assert user is not None
        assert user.username == "testuser"
        
        # Get non-existent user
        user = user_manager.get_user("nonexistent")
        assert user is None
    
    def test_list_users(self, user_manager):
        """Test listing all users."""
        # Create multiple users
        user_manager.create_user("user1", "user1@example.com")
        user_manager.create_user("user2", "user2@example.com")
        
        # List users
        users = user_manager.list_users()
        assert len(users) == 3  # admin + user1 + user2
        
        usernames = [user.username for user in users]
        assert "admin" in usernames
        assert "user1" in usernames
        assert "user2" in usernames
    
    def test_update_user(self, user_manager):
        """Test updating user information."""
        # Create user
        user_manager.create_user("testuser", "test@example.com")
        
        # Update user
        success = user_manager.update_user("testuser", email="updated@example.com", role=UserRole.VIEWER)
        assert success is True
        
        # Verify update
        user = user_manager.get_user("testuser")
        assert user.email == "updated@example.com"
        assert user.role == UserRole.VIEWER
    
    def test_update_nonexistent_user(self, user_manager):
        """Test updating non-existent user."""
        success = user_manager.update_user("nonexistent", email="test@example.com")
        assert success is False
    
    def test_delete_user(self, user_manager):
        """Test user deletion."""
        # Create user
        user_manager.create_user("testuser", "test@example.com")
        
        # Delete user
        success = user_manager.delete_user("testuser")
        assert success is True
        
        # Verify deletion
        user = user_manager.get_user("testuser")
        assert user is None
    
    def test_delete_admin_user(self, user_manager):
        """Test that admin user cannot be deleted."""
        success = user_manager.delete_user("admin")
        assert success is False
        
        # Verify admin still exists
        user = user_manager.get_user("admin")
        assert user is not None
        assert user.username == "admin"
    
    def test_create_project(self, user_manager):
        """Test project creation."""
        # Create user
        user_manager.create_user("testuser", "test@example.com")
        
        # Create project
        project_id = user_manager.create_project("Test Project", "testuser", "Test description")
        assert project_id is not None
        
        # Verify project was created
        project = user_manager.get_project(project_id)
        assert project is not None
        assert project.name == "Test Project"
        assert project.owner == "testuser"
        assert project.description == "Test description"
        
        # Verify project directory was created
        project_dir = Path(f"projects/{project_id}")
        assert project_dir.exists()
        assert (project_dir / "data").exists()
        assert (project_dir / "config").exists()
        assert (project_dir / "models").exists()
        assert (project_dir / "experiments").exists()
        assert (project_dir / "deployment").exists()
        assert (project_dir / "monitoring").exists()
        assert (project_dir / "results").exists()
        
        # Verify config file was created
        config_file = project_dir / "config" / "project_config.yaml"
        assert config_file.exists()
        
        # Clean up
        shutil.rmtree(project_dir)
    
    def test_create_project_nonexistent_owner(self, user_manager):
        """Test creating project with non-existent owner."""
        project_id = user_manager.create_project("Test Project", "nonexistent", "Test description")
        assert project_id is None
    
    def test_get_project(self, user_manager):
        """Test getting project by ID."""
        # Create user and project
        user_manager.create_user("testuser", "test@example.com")
        project_id = user_manager.create_project("Test Project", "testuser")
        
        # Get project
        project = user_manager.get_project(project_id)
        assert project is not None
        assert project.project_id == project_id
        
        # Get non-existent project
        project = user_manager.get_project("nonexistent")
        assert project is None
        
        # Clean up
        project_dir = Path(f"projects/{project_id}")
        if project_dir.exists():
            shutil.rmtree(project_dir)
    
    def test_list_user_projects(self, user_manager):
        """Test listing user's projects."""
        # Create users
        user_manager.create_user("user1", "user1@example.com")
        user_manager.create_user("user2", "user2@example.com")
        
        # Create projects
        project1_id = user_manager.create_project("Project 1", "user1")
        project2_id = user_manager.create_project("Project 2", "user1")
        project3_id = user_manager.create_project("Project 3", "user2")
        
        # List user1's projects
        user1_projects = user_manager.list_user_projects("user1")
        assert len(user1_projects) == 2
        project_names = [p.name for p in user1_projects]
        assert "Project 1" in project_names
        assert "Project 2" in project_names
        
        # List user2's projects
        user2_projects = user_manager.list_user_projects("user2")
        assert len(user2_projects) == 1
        assert user2_projects[0].name == "Project 3"
        
        # Clean up
        for project_id in [project1_id, project2_id, project3_id]:
            project_dir = Path(f"projects/{project_id}")
            if project_dir.exists():
                shutil.rmtree(project_dir)
    
    def test_list_all_projects(self, user_manager):
        """Test listing all projects."""
        # Create users and projects
        user_manager.create_user("user1", "user1@example.com")
        user_manager.create_user("user2", "user2@example.com")
        
        user_manager.create_project("Project 1", "user1")
        user_manager.create_project("Project 2", "user2")
        
        # List all projects
        all_projects = user_manager.list_all_projects()
        assert len(all_projects) == 2
        
        project_names = [p.name for p in all_projects]
        assert "Project 1" in project_names
        assert "Project 2" in project_names
        
        # Clean up
        for project in all_projects:
            project_dir = Path(f"projects/{project.project_id}")
            if project_dir.exists():
                shutil.rmtree(project_dir)
    
    def test_update_project(self, user_manager):
        """Test updating project information."""
        # Create user and project
        user_manager.create_user("testuser", "test@example.com")
        project_id = user_manager.create_project("Test Project", "testuser", "Original description")
        
        # Update project
        success = user_manager.update_project(project_id, name="Updated Project", description="Updated description")
        assert success is True
        
        # Verify update
        project = user_manager.get_project(project_id)
        assert project.name == "Updated Project"
        assert project.description == "Updated description"
        
        # Clean up
        project_dir = Path(f"projects/{project_id}")
        if project_dir.exists():
            shutil.rmtree(project_dir)
    
    def test_update_nonexistent_project(self, user_manager):
        """Test updating non-existent project."""
        success = user_manager.update_project("nonexistent", name="Updated")
        assert success is False
    
    def test_delete_project(self, user_manager):
        """Test project deletion."""
        # Create user and project
        user_manager.create_user("testuser", "test@example.com")
        project_id = user_manager.create_project("Test Project", "testuser")
        
        # Verify project directory exists
        project_dir = Path(f"projects/{project_id}")
        assert project_dir.exists()
        
        # Delete project
        success = user_manager.delete_project(project_id, "testuser")
        assert success is True
        
        # Verify project was deleted
        project = user_manager.get_project(project_id)
        assert project is None
        
        # Verify project directory was removed
        assert not project_dir.exists()
    
    def test_delete_project_unauthorized(self, user_manager):
        """Test deleting project without authorization."""
        # Create users and project
        user_manager.create_user("user1", "user1@example.com")
        user_manager.create_user("user2", "user2@example.com")
        project_id = user_manager.create_project("Test Project", "user1")
        
        # Try to delete with different user
        success = user_manager.delete_project(project_id, "user2")
        assert success is False
        
        # Verify project still exists
        project = user_manager.get_project(project_id)
        assert project is not None
        
        # Clean up
        project_dir = Path(f"projects/{project_id}")
        if project_dir.exists():
            shutil.rmtree(project_dir)
    
    def test_check_permission(self, user_manager):
        """Test permission checking."""
        # Create users and project
        user_manager.create_user("user1", "user1@example.com", UserRole.USER)
        user_manager.create_user("user2", "user2@example.com", UserRole.VIEWER)
        project_id = user_manager.create_project("Test Project", "user1")
        
        # Admin can do everything
        assert user_manager.check_permission("admin", project_id, "read") is True
        assert user_manager.check_permission("admin", project_id, "write") is True
        assert user_manager.check_permission("admin", project_id, "delete") is True
        
        # Project owner can do everything
        assert user_manager.check_permission("user1", project_id, "read") is True
        assert user_manager.check_permission("user1", project_id, "write") is True
        assert user_manager.check_permission("user1", project_id, "delete") is True
        
        # Viewer can only read
        assert user_manager.check_permission("user2", project_id, "read") is True
        assert user_manager.check_permission("user2", project_id, "write") is False
        assert user_manager.check_permission("user2", project_id, "delete") is False
        
        # User cannot access other user's project
        user_manager.create_user("user3", "user3@example.com", UserRole.USER)
        assert user_manager.check_permission("user3", project_id, "read") is False
        assert user_manager.check_permission("user3", project_id, "write") is False
        
        # Clean up
        project_dir = Path(f"projects/{project_id}")
        if project_dir.exists():
            shutil.rmtree(project_dir)
    
    def test_get_user_stats(self, user_manager):
        """Test getting user statistics."""
        # Create user and projects
        user_manager.create_user("testuser", "test@example.com")
        project1_id = user_manager.create_project("Project 1", "testuser")
        project2_id = user_manager.create_project("Project 2", "testuser")
        
        # Get user stats
        stats = user_manager.get_user_stats("testuser")
        assert stats["username"] == "testuser"
        assert stats["role"] == "user"
        assert stats["total_projects"] == 2
        assert stats["active_projects"] == 2
        assert stats["total_models"] == 0
        assert stats["total_experiments"] == 0
        
        # Get stats for non-existent user
        stats = user_manager.get_user_stats("nonexistent")
        assert stats == {}
        
        # Clean up
        for project_id in [project1_id, project2_id]:
            project_dir = Path(f"projects/{project_id}")
            if project_dir.exists():
                shutil.rmtree(project_dir)
    
    def test_get_system_stats(self, user_manager):
        """Test getting system statistics."""
        # Create users and projects
        user_manager.create_user("user1", "user1@example.com", UserRole.USER)
        user_manager.create_user("user2", "user2@example.com", UserRole.VIEWER)
        user_manager.create_project("Project 1", "user1")
        user_manager.create_project("Project 2", "user2")
        
        # Get system stats
        stats = user_manager.get_system_stats()
        assert stats["total_users"] == 3  # admin + user1 + user2
        assert stats["total_projects"] == 2
        assert stats["active_users"] == 3
        assert stats["total_models"] == 0
        assert stats["total_experiments"] == 0
        assert stats["users_by_role"]["admin"] == 1
        assert stats["users_by_role"]["user"] == 1
        assert stats["users_by_role"]["viewer"] == 1
        
        # Clean up
        for project in user_manager.list_all_projects():
            project_dir = Path(f"projects/{project.project_id}")
            if project_dir.exists():
                shutil.rmtree(project_dir)
    
    def test_data_persistence(self, temp_data_dir):
        """Test that data persists between user manager instances."""
        # Create first user manager and add data
        user_manager1 = UserManager(data_dir=temp_data_dir)
        user_manager1.create_user("testuser", "test@example.com")
        project_id = user_manager1.create_project("Test Project", "testuser")
        
        # Create second user manager and verify data
        user_manager2 = UserManager(data_dir=temp_data_dir)
        user = user_manager2.get_user("testuser")
        assert user is not None
        assert user.username == "testuser"
        
        project = user_manager2.get_project(project_id)
        assert project is not None
        assert project.name == "Test Project"
        
        # Clean up
        project_dir = Path(f"projects/{project_id}")
        if project_dir.exists():
            shutil.rmtree(project_dir)

if __name__ == "__main__":
    pytest.main([__file__]) 