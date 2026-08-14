import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from config.config_script import load_dataframe, setup_logger
from src.preprocessor import preprocess_regression

if __name__ == "__main__":
    logger = setup_logger('pipeline_run')
    logger.info("Running pipeline.py orchestration...")
    try:
        df_regression, df_classification = preprocess_regression().fit()
    
        X_train, X_test, y_train, y_test, indices_train, indices_test, df_meta, feature_cols, categorical_features, boolean_features, numeric_features = preprocess_regression().fit_transform(df_regression)
        logger.info("Orchestration completed successfully.")
    except Exception as e:
        logger.exception(f"Unhandled exception during direct run: {e}")