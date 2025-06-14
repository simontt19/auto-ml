"""
User Management System
Multi-user support with authentication, authorization, and project isolation.
"""

import os
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

@dataclass
class User:
    """User data structure."""
    username: str
    email: str
    role: UserRole
    created_at: str
    last_login: Optional[str] = None
    is_active: bool = True
    projects: List[str] = None
    
    def __post_init__(self):
        if self.projects is None:
            self.projects = []

@dataclass
class Project:
    """Project data structure."""
    project_id: str
    name: str
    owner: str
    description: str
    created_at: str
    updated_at: str
    config: Dict[str, Any] = None
    models: List[str] = None
    experiments: List[str] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.models is None:
            self.models = []
        if self.experiments is None:
            self.experiments = []

class UserManager:
    """
    User management system for multi-user support.
    
    Features:
    - User authentication and authorization
    - Project isolation and management
    - Role-based access control
    - User session management
    """
    
    def __init__(self, data_dir: str = "data/users"):
        """
        Initialize user management system.
        
        Args:
            data_dir (str): Directory to store user and project data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.users_file = self.data_dir / "users.json"
        self.projects_file = self.data_dir / "projects.json"
        self.sessions_file = self.data_dir / "sessions.json"
        
        # In-memory storage (for performance)
        self.users: Dict[str, User] = {}
        self.projects: Dict[str, Project] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_data()
        
        # Create default admin user if none exists
        if not self.users:
            self._create_default_admin()
    
    def _load_data(self) -> None:
        """Load users, projects, and sessions from files."""
        try:
            # Load users
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                    for username, user_data in users_data.items():
                        user_data['role'] = UserRole(user_data['role'])
                        self.users[username] = User(**user_data)
            
            # Load projects
            if self.projects_file.exists():
                with open(self.projects_file, 'r') as f:
                    projects_data = json.load(f)
                    for project_id, project_data in projects_data.items():
                        self.projects[project_id] = Project(**project_data)
            
            # Load sessions
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    self.sessions = json.load(f)
            
            logger.info(f"Loaded {len(self.users)} users and {len(self.projects)} projects")
            
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
    
    def _save_data(self) -> None:
        """Save users, projects, and sessions to files."""
        try:
            # Save users (convert UserRole to string for JSON serialization)
            users_data = {}
            for username, user in self.users.items():
                user_dict = asdict(user)
                user_dict['role'] = user_dict['role'].value  # Convert enum to string
                users_data[username] = user_dict
            
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
            
            # Save projects
            projects_data = {
                project_id: asdict(project) for project_id, project in self.projects.items()
            }
            with open(self.projects_file, 'w') as f:
                json.dump(projects_data, f, indent=2)
            
            # Save sessions
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
            
            logger.info("User data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin_user = User(
            username="admin",
            email="admin@automl.local",
            role=UserRole.ADMIN,
            created_at=datetime.now().isoformat()
        )
        self.users["admin"] = admin_user
        self._save_data()
        logger.info("Created default admin user: admin")
    
    def create_user(self, username: str, email: str, role: UserRole = UserRole.USER) -> bool:
        """
        Create a new user.
        
        Args:
            username (str): Username
            email (str): Email address
            role (UserRole): User role
            
        Returns:
            bool: True if user created successfully
        """
        if username in self.users:
            logger.warning(f"User {username} already exists")
            return False
        
        user = User(
            username=username,
            email=email,
            role=role,
            created_at=datetime.now().isoformat()
        )
        
        self.users[username] = user
        self._save_data()
        
        logger.info(f"Created user: {username} with role {role.value}")
        return True
    
    def get_user(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username (str): Username
            
        Returns:
            Optional[User]: User object or None if not found
        """
        return self.users.get(username)
    
    def list_users(self) -> List[User]:
        """
        List all users.
        
        Returns:
            List[User]: List of all users
        """
        return list(self.users.values())
    
    def update_user(self, username: str, **kwargs) -> bool:
        """
        Update user information.
        
        Args:
            username (str): Username
            **kwargs: Fields to update
            
        Returns:
            bool: True if user updated successfully
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return False
        
        user = self.users[username]
        
        # Update allowed fields
        allowed_fields = ['email', 'role', 'is_active', 'last_login']
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(user, field, value)
        
        self._save_data()
        logger.info(f"Updated user: {username}")
        return True
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user.
        
        Args:
            username (str): Username
            
        Returns:
            bool: True if user deleted successfully
        """
        if username not in self.users:
            logger.warning(f"User {username} not found")
            return False
        
        # Don't allow deletion of admin user
        if username == "admin":
            logger.warning("Cannot delete admin user")
            return False
        
        # Remove user from projects
        for project in self.projects.values():
            if project.owner == username:
                project.owner = "admin"  # Transfer to admin
        
        del self.users[username]
        self._save_data()
        
        logger.info(f"Deleted user: {username}")
        return True
    
    def create_project(self, name: str, owner: str, description: str = "") -> Optional[str]:
        """
        Create a new project.
        
        Args:
            name (str): Project name
            owner (str): Project owner username
            description (str): Project description
            
        Returns:
            Optional[str]: Project ID if created successfully
        """
        if owner not in self.users:
            logger.warning(f"Owner {owner} not found")
            return None
        
        # Generate unique project ID
        project_id = f"{owner}_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_id = project_id.replace(" ", "_").lower()
        
        # Check if project already exists
        if project_id in self.projects:
            logger.warning(f"Project {project_id} already exists")
            return None
        
        project = Project(
            project_id=project_id,
            name=name,
            owner=owner,
            description=description,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.projects[project_id] = project
        
        # Add project to user's project list
        if owner in self.users:
            self.users[owner].projects.append(project_id)
        
        self._save_data()
        
        # Create project directory structure
        self._create_project_structure(project_id)
        
        logger.info(f"Created project: {project_id} for user {owner}")
        return project_id
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get project by ID.
        
        Args:
            project_id (str): Project ID
            
        Returns:
            Optional[Project]: Project object or None if not found
        """
        return self.projects.get(project_id)
    
    def list_user_projects(self, username: str) -> List[Project]:
        """
        List projects owned by a user.
        
        Args:
            username (str): Username
            
        Returns:
            List[Project]: List of user's projects
        """
        user_projects = []
        for project in self.projects.values():
            if project.owner == username:
                user_projects.append(project)
        return user_projects
    
    def list_all_projects(self) -> List[Project]:
        """
        List all projects.
        
        Returns:
            List[Project]: List of all projects
        """
        return list(self.projects.values())
    
    def update_project(self, project_id: str, **kwargs) -> bool:
        """
        Update project information.
        
        Args:
            project_id (str): Project ID
            **kwargs: Fields to update
            
        Returns:
            bool: True if project updated successfully
        """
        if project_id not in self.projects:
            logger.warning(f"Project {project_id} not found")
            return False
        
        project = self.projects[project_id]
        
        # Update allowed fields
        allowed_fields = ['name', 'description', 'config']
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(project, field, value)
        
        project.updated_at = datetime.now().isoformat()
        self._save_data()
        
        logger.info(f"Updated project: {project_id}")
        return True
    
    def delete_project(self, project_id: str, username: str) -> bool:
        """
        Delete a project.
        
        Args:
            project_id (str): Project ID
            username (str): Username requesting deletion
            
        Returns:
            bool: True if project deleted successfully
        """
        if project_id not in self.projects:
            logger.warning(f"Project {project_id} not found")
            return False
        
        project = self.projects[project_id]
        
        # Check permissions
        user = self.users.get(username)
        if not user:
            logger.warning(f"User {username} not found")
            return False
        
        if project.owner != username and user.role != UserRole.ADMIN:
            logger.warning(f"User {username} not authorized to delete project {project_id}")
            return False
        
        # Remove project from user's project list
        if project.owner in self.users:
            if project_id in self.users[project.owner].projects:
                self.users[project.owner].projects.remove(project_id)
        
        del self.projects[project_id]
        self._save_data()
        
        # Remove project directory
        self._remove_project_structure(project_id)
        
        logger.info(f"Deleted project: {project_id}")
        return True
    
    def _create_project_structure(self, project_id: str) -> None:
        """Create project directory structure."""
        project_dir = Path(f"projects/{project_id}")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "config").mkdir(exist_ok=True)
        (project_dir / "models").mkdir(exist_ok=True)
        (project_dir / "experiments").mkdir(exist_ok=True)
        (project_dir / "deployment").mkdir(exist_ok=True)
        (project_dir / "monitoring").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)
        
        # Create default config file
        default_config = {
            "project_id": project_id,
            "created_at": datetime.now().isoformat(),
            "data": {
                "source_path": "",
                "target_column": "",
                "validation_split": 0.2
            },
            "features": {
                "engineering_type": "standard",
                "categorical_columns": [],
                "numerical_columns": []
            },
            "model": {
                "task_type": "classification",
                "algorithms": ["logistic_regression", "random_forest", "lightgbm"],
                "hyperparameter_optimization": True
            },
            "monitoring": {
                "drift_threshold": 0.05,
                "performance_threshold": 0.1,
                "alert_on_drift": True
            }
        }
        
        config_file = project_dir / "config" / "project_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created project structure for {project_id}")
    
    def _remove_project_structure(self, project_id: str) -> None:
        """Remove project directory structure."""
        import shutil
        project_dir = Path(f"projects/{project_id}")
        if project_dir.exists():
            shutil.rmtree(project_dir)
            logger.info(f"Removed project structure for {project_id}")
    
    def check_permission(self, username: str, project_id: str, action: str) -> bool:
        """
        Check if user has permission for an action on a project.
        
        Args:
            username (str): Username
            project_id (str): Project ID
            action (str): Action to check (read, write, delete, admin)
            
        Returns:
            bool: True if user has permission
        """
        if project_id not in self.projects:
            return False
        
        if username not in self.users:
            return False
        
        user = self.users[username]
        project = self.projects[project_id]
        
        # Admin can do everything
        if user.role == UserRole.ADMIN:
            return True
        
        # Project owner can do everything
        if project.owner == username:
            return True
        
        # Viewer can only read
        if user.role == UserRole.VIEWER and action == "read":
            return True
        
        # User can read and write their own projects
        if user.role == UserRole.USER and action in ["read", "write"]:
            return project.owner == username
        
        return False
    
    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Args:
            username (str): Username
            
        Returns:
            Dict[str, Any]: User statistics
        """
        if username not in self.users:
            return {}
        
        user = self.users[username]
        user_projects = self.list_user_projects(username)
        
        stats = {
            "username": username,
            "role": user.role.value,
            "created_at": user.created_at,
            "last_login": user.last_login,
            "total_projects": len(user_projects),
            "active_projects": len([p for p in user_projects if p.owner == username]),
            "total_models": sum(len(p.models) for p in user_projects),
            "total_experiments": sum(len(p.experiments) for p in user_projects)
        }
        
        return stats
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide statistics.
        
        Args:
            Dict[str, Any]: System statistics
        """
        stats = {
            "total_users": len(self.users),
            "total_projects": len(self.projects),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "total_models": sum(len(p.models) for p in self.projects.values()),
            "total_experiments": sum(len(p.experiments) for p in self.projects.values()),
            "users_by_role": {
                role.value: len([u for u in self.users.values() if u.role == role])
                for role in UserRole
            }
        }
        
        return stats 