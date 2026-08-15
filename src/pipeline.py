import pandas as pd
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from config.config_script import load_dataframe, setup_logger, X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, Y_PRED_PATH, DF_META_PATH
from src.preprocessor import preprocess_regression
from src.train import Pipeline_train
from src.evaluate import Pipeline_evaluate

if __name__ == "__main__":
    logger = setup_logger('preprocessor_run')
    logger.info("Running preprocessor.py directly...")
    try:
        df_regression, df_classification = preprocess_regression().fit()
        
        X_train, X_test, y_train, y_test, indices_train, indices_test, df_meta, feature_cols, categorical_features, boolean_features, numeric_features = preprocess_regression().fit_transform(df_regression)

        pd.DataFrame(X_train).to_parquet(X_TRAIN_PATH, index=False)
        pd.DataFrame(X_test).to_parquet(X_TEST_PATH, index=False)
        pd.DataFrame(y_train).to_parquet(Y_TRAIN_PATH, index=False)
        pd.DataFrame(y_test).to_parquet(Y_TEST_PATH, index=False)
        pd.DataFrame(df_meta).to_parquet(DF_META_PATH, index=False)

        logger.info("Preprocessed data saved to artifacts/preprocessed directory.")
    except Exception as e:
        logger.exception(f"Unhandled exception during direct run: {e}")

    pipeline_train = Pipeline_train()
    pipeline_train.fit_model_regression(verbose=0)