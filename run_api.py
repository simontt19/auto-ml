#!/usr/bin/env python3
"""
Wrapper script to run the Auto ML FastAPI server with correct PYTHONPATH.
"""
import sys
import os
from pathlib import Path

# Ensure the project root and auto_ml are in the path
project_root = Path(__file__).parent.resolve()
auto_ml_dir = project_root / 'auto_ml'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(auto_ml_dir))

from auto_ml.deployment.api.simple_api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 