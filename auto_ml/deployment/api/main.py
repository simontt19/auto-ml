#!/usr/bin/env python3
"""
Main entry point for the Auto ML API server.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from auto_ml.deployment.api.model_api import ModelAPI
import uvicorn

def main():
    """Start the FastAPI server."""
    # Initialize the API
    api = ModelAPI(models_base_dir="projects")
    
    # Get the FastAPI app
    app = api.get_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 