"""
Inference Pipeline Module
Loads a saved model and feature pipeline, applies them to new data, and outputs predictions.
Supports both programmatic and CLI batch inference.
"""

import argparse
import logging
import pandas as pd
from model_persistence import ModelPersistence
from typing import Optional, Union
import sys

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO") -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    for handler in logging.root.handlers:
        handler.setLevel(numeric_level)

def run_inference(
    input_data: Union[str, pd.DataFrame],
    model_name: str,
    version_id: Optional[str] = None,
    output_path: Optional[str] = None,
    log_level: str = "INFO"
) -> pd.DataFrame:
    """
    Run inference using a saved model and feature pipeline.
    Args:
        input_data (str or pd.DataFrame): Path to CSV file or DataFrame with new data
        model_name (str): Name of the model to use
        version_id (str, optional): Specific model version (uses latest if None)
        output_path (str, optional): Path to save predictions CSV
        log_level (str): Logging level
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    setup_logging(log_level)
    logger.info(f"Loading model '{model_name}' (version: {version_id or 'latest'})...")
    mp = ModelPersistence()
    model, metadata = mp.load_model(model_name, version_id)
    pipeline = mp.load_feature_pipeline(model_name, version_id)
    feature_names = metadata.get("feature_names")
    
    # Load input data
    if isinstance(input_data, str):
        logger.info(f"Reading input data from {input_data}...")
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()
    
    # Apply feature pipeline if available
    if pipeline is not None:
        logger.info("Applying feature engineering pipeline...")
        # Try to infer categorical/numerical columns if possible
        if hasattr(pipeline, 'transform'):
            # Try to get columns from metadata or pipeline
            cat_cols = getattr(pipeline, 'cat_cols', None)
            num_cols = getattr(pipeline, 'num_cols', None)
            if cat_cols is not None and num_cols is not None:
                df_processed = pipeline.transform(df, cat_cols, num_cols)
            else:
                df_processed = pipeline.transform(df)
        else:
            logger.warning("Feature pipeline does not have a 'transform' method. Skipping transformation.")
            df_processed = df
    else:
        logger.warning("No feature pipeline found. Using raw input data.")
        df_processed = df
    
    # Ensure feature columns match
    if feature_names is not None:
        missing = set(feature_names) - set(df_processed.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X = df_processed[feature_names].fillna(0)
    else:
        X = df_processed.fillna(0)
    
    # Predict
    logger.info("Running predictions...")
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        y_pred_proba = None
    
    # Prepare output
    result = df.copy()
    result['prediction'] = y_pred
    if y_pred_proba is not None:
        result['prediction_proba'] = y_pred_proba
    
    # Save if requested
    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    else:
        logger.info("Inference completed. Returning DataFrame.")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference using a saved model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model-name", type=str, required=True, help="Model name (e.g. adult_income_lightgbm)")
    parser.add_argument("--version-id", type=str, help="Model version ID (default: latest)")
    parser.add_argument("--output", type=str, help="Path to save predictions CSV")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    
    run_inference(
        input_data=args.input,
        model_name=args.model_name,
        version_id=args.version_id,
        output_path=args.output,
        log_level=args.log_level
    ) 