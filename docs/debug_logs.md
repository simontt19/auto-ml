## [$(date '+%Y-%m-%d %H:%M:%S')] Troubleshooting: Always Use venv for Local Testing

- Issue: Local testing and server runs were using the system Python (3.13) instead of the project venv (likely 3.11), causing import and compatibility issues.
- Solution: Always activate the venv before running any Python or Uvicorn commands:
  ```bash
  source venv/bin/activate
  venv/bin/python ...
  venv/bin/uvicorn ...
  ```
- All dependencies must be installed into the venv:
  ```bash
  venv/bin/pip install -r requirements.txt
  ```
- This ensures local runs match the deployment environment and avoids system/venv conflicts.
