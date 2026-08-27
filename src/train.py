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
from config.config_script import numeric_features, categorical_features, boolean_features, feature_cols, target_col
from evaluate import Pipeline_evaluate

from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings('ignore')

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

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.boolean_features = boolean_features

        self.feature_cols = feature_cols
        self.target_col = target_col

        self.model_regressor = LGBMRegressor(
            random_state=42, 
            objective='regression_l1', 
            learning_rate=0.05, 
            force_row_wise = True, 
            reg_alpha=0.1, 
            reg_lambda=1, 
            n_jobs=-1,
            verbosity=-1)

        self.model_classifier = None

    def transformer_regression(self):
        self.column_transformer = ColumnTransformer(
            transformers = [
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output = False), self.categorical_features),
                ('bool', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), self.boolean_features)
            ], remainder = 'passthrough'
        )
        return self.column_transformer

    def hyperparameter_tuning(self, pipeline):
        self.logger.info("Initiating Hyperparameter tuning:")
        try:
            rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
            param_grid = {
                'regressor__regressor__num_leaves': [7, 15],          # Diturunkan agar tree tidak terlalu dalam
                'regressor__regressor__min_child_samples': [20, 40],   # Dinaikkan agar leaf butuh sampel lebih banyak
                'regressor__regressor__learning_rate': [0.01, 0.03, 0.05],
                'regressor__regressor__reg_alpha': [0.1, 1.0],        # Paksa regularisasi L1
                'regressor__regressor__reg_lambda': [0.1, 1.0],        # Paksa regularisasi L2
                'regressor__regressor__max_depth': [-1, 3, 5],
                'regressor__regressor__importance_type': ['gain', 'split']
            }

            scorers = {
                'r2': make_scorer(r2_score),
                'mae': make_scorer(mean_absolute_error, greater_is_better=False), # Negatif karena Sklearn mengoptimalkan loss
                'smape': make_scorer(self.calculate_smape)
            }

            grid_search = GridSearchCV(
                estimator= pipeline,
                param_grid = param_grid,
                cv = rkf,
                scoring=scorers,
                refit = 'r2',
                n_jobs = -1,
                verbose = 1
            )

            grid_search.fit(self.X_train_regression, self.y_train_regression)
            self.logger.info(f"Best parameters config:", grid_search.best_params_)
            self.logger.info(f"Best CV after hyperparameter tuning -- r2 score: {grid_search.best_score_:.3f}")
            self.logger.info(f"[Hyperparameter Tuning Succesful!]")
        except Exception as e:
            self.logger.error(f"Error tuning pipeline regression model {e}")


        return grid_search.best_estimator_, grid_search.cv_results_, grid_search.best_index_, grid_search
    
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

        self.logger.info(f"=" * 50)
        self.pipeline_regression, self.cv_results, self.best_index, self.grid_search = self.hyperparameter_tuning(self.pipeline_regression)
        self.y_pred_regression = self.pipeline_regression.predict(self.X_test_regression)
        
        self.logger.info(f"Saving predictions...")
        try:
            pd.DataFrame(self.y_pred_regression).to_parquet(Y_PRED_PATH, index=False)
            self.logger.info(f"Predictions saved to {Y_PRED_PATH}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {e}")
            raise

        self.logger.info("Calling evaluation.py through super().__init__...")
        try:
            super().__init__()
        except Exception as e:
            self.logger.error(f"Error calling evaluation.py through super().__init__(): {e}")
            raise

        self.logger.info("Starting evaluation pipeline.")

        self.evaluate = Pipeline_evaluate()
        self.evaluate.evaluate_regression(verbose, cv_results=self.cv_results, best_index=self.best_index, pipeline=self.pipeline_regression, grid_search = self.grid_search)

        joblib.dump(self.pipeline_regression, MODEL_REGRESSION_PATH, compress = 3)
        self.logger.info(f"Final regression pipeline saved to {MODEL_REGRESSION_PATH}")

    def transformer_classification(self):
        pass
    
    def fit_model_classification(self):
        pass

if __name__ == "__main__":
    pipeline_train = Pipeline_train()
    pipeline_train.fit_model_regression(verbose=1)

        
        