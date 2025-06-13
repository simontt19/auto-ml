# Debug Logs & Troubleshooting

## 2024-06-14: Error Handling and Logging Improvements

- Added try/except blocks to all main methods in `feature_engineering.py` and `model_training.py`.
- All exceptions are now logged with `logger.error` and include the full traceback for easier debugging.
- Example log entry (from pipeline.log):

```
ERROR:feature_engineering:Error in fit_transform: ValueError: could not convert string to float: 'Private'
Traceback (most recent call last):
  File "feature_engineering.py", line 25, in fit_transform
    ...
ValueError: could not convert string to float: 'Private'
```

- This ensures that any failure in feature engineering or model training is immediately visible in the logs, with a clear traceback for diagnosis.

## Next Steps

- Continue to log any issues, fixes, and debugging insights here as the project evolves.
