import sys
import os
import pandas as pd
import pyarrow
import fastparquet
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from config.config_script import load_dataframe, setup_logger, X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, Y_PRED_PATH, DF_META_PATH, MODEL_REGRESSION_PATH
from evaluate import Pipeline_evaluate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

class Pipeline_train(Pipeline_evaluate):
    def __init__(self):
        self.logger = setup_logger('train_run')
        try:
            self.X_train_regression = pd.read_parquet(X_TRAIN_PATH)
            self.X_test_regression = pd.read_parquet(X_TEST_PATH)
            self.y_train_regression = pd.read_parquet(Y_TRAIN_PATH)
            self.y_test_regression = pd.read_parquet(Y_TEST_PATH)
            self.df_meta_regression = pd.read_parquet(DF_META_PATH)
            self.logger.info("Preprocessed data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading preprocessed data: {e}, make sure to run preprocessor.py first!")
            raise 

        self.numeric_features = ['title_length', 'title_word_count', 'volume_count', 'text_risk_score', 'condition_encoded', 'volume_tier_encoded']
        self.categorical_features = ['currency', 'seller_location']
        self.boolean_features = ['is_boxset', 'is_special_edition', 'boxset_side_story_edition_included', 'standalone_side_story_edition']

        self.feature_cols = self.numeric_features + self.categorical_features + self.boolean_features
        self.target_col = 'price'

        self.model_regressor = XGBRegressor(
            random_state=42, 
            tree_method = 'hist', 
            objective='reg:absoluteerror', 
            n_jobs = -1
        )

        self.model_classifier = None

    def transformer_regression(self):
        self.column_transformer = ColumnTransformer(
            transformers = [
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output = False), self.categorical_features),
                ('bool', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), self.boolean_features)
            ], remainder = 'passthrough'
        )
        return self.column_transformer
    
    def fit_model_regression(self, verbose:int):
        self.logger.info("Starting comprehensive modelling pipeline.")

        self.transformed_model_regression = TransformedTargetRegressor(
            regressor=self.model_regressor,
            func=np.log1p,       
            inverse_func=np.expm1 
        )
        
        self.pipeline_regression = Pipeline(steps = [
            ('preprocessor', self.transformer_regression()),
            ('scale', RobustScaler(with_centering=False)), 
            ('regressor', self.transformed_model_regression)
        ])
        
        self.logger.info("Training regression model...")
        try:
            self.pipeline_regression.fit(self.X_train_regression, self.y_train_regression)
            self.y_pred_regression = self.pipeline_regression.predict(self.X_test_regression)
        except Exception as e:
            self.logger.error(f"Error training regression model: {e}")
            raise
        
        self.logger.info(f"Saving predictions...")
        try:
            pd.DataFrame(self.y_pred_regression).to_parquet(Y_PRED_PATH, index=False)
            self.logger.info(f"Predictions saved to {Y_PRED_PATH}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {e}")
            raise

        self.logger.info("Calling super().__init__...")
        try:
            super().__init__()
        except Exception as e:
            self.logger.error(f"Error calling super().__init__(): {e}")
            raise

        self.logger.info("Starting evaluation pipeline.")
        self.evaluate = Pipeline_evaluate()
        self.evaluate.evaluate_regression(verbose, pipeline=self.pipeline_regression)

        joblib.dump(self.pipeline_regression, MODEL_REGRESSION_PATH, compress = 3)
        self.logger.info(f"Final regression pipeline saved to {MODEL_REGRESSION_PATH}")

    def transformer_classification(self):
        pass
    
    def fit_model_classification(self):
        pass

if __name__ == "__main__":
    pipeline_train = Pipeline_train()
    pipeline_train.fit_model_regression(verbose=1)

        
        