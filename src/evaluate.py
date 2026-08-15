import pandas as pd
import pyarrow
import fastparquet
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from config.config_script import load_dataframe, setup_logger, X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, Y_PRED_PATH, DF_META_PATH, EVAL_SUMMARY_REGRESSION_PATH, EVAL_DATAFRAME_REGRESSION_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

class Pipeline_evaluate():
    def __init__(self):
        self.logger = setup_logger('evaluate')
        self.logger.info("[evaluate.py] Reading test & prediction data...")
        try:
            self.X_train_regression = pd.read_parquet(X_TRAIN_PATH)
            self.X_test_regression = pd.read_parquet(X_TEST_PATH)
            self.y_train_regression = pd.read_parquet(Y_TRAIN_PATH)
            self.y_test_regression = pd.read_parquet(Y_TEST_PATH)
            self.y_pred_regression = pd.read_parquet(Y_PRED_PATH)
            self.df_meta_regression = pd.read_parquet(DF_META_PATH)
            self.indices_train = pd.read_parquet(X_TRAIN_PATH).index
            self.indices_test = pd.read_parquet(X_TEST_PATH).index
            self.logger.info("[evaluate.py] Data loaded successfully.")
        except Exception as e:
            self.logger.error(f"[evaluate.py] Error loading data: {e}, make sure you run preprocessor.py & train.py first!")
            raise
        
    def calculate_smape(self):
        y_true, y_pred = np.array(self.y_test_regression), np.array(self.y_pred_regression)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        epsilon = 1e-10
        denominator = np.where(denominator == 0, epsilon, denominator)
        return np.mean(np.abs(y_pred - y_true) / denominator) * 100

    def output_json_regression(self):
        self.logger.info(f"Saving Model Evaluation Summary...")
        try:
            self.metrics_regression.to_json(EVAL_SUMMARY_REGRESSION_PATH, orient='records', indent=4)
            self.logger.info(f"Evaluation metrics saved to {EVAL_SUMMARY_REGRESSION_PATH}")

            self.logger.info(f"Printing evaluation report into JSON format...")
            self.results_regression.to_json(EVAL_DATAFRAME_REGRESSION_PATH, orient='records', indent=4)
            self.logger.info(f"Evaluation dataframe saved to {EVAL_DATAFRAME_REGRESSION_PATH}")
        except Exception as e:
            self.logger.error(f"Error saving JSON output: {e}")
            raise
        
    def evaluate_regression(self, verbose = 0, pipeline = None):
        self.logger.info("Evaluating model performance.")
        try:
            if pipeline is not None:
                if hasattr(pipeline, 'steps'):
                    # Ambil step terakhir (dalam kasus Anda: 'regressor')
                    final_estimator = pipeline.steps[-1][1]
                    
                    # Cek apakah model dibungkus oleh TransformedTargetRegressor
                    if type(final_estimator).__name__ == "TransformedTargetRegressor":
                        # Ekstrak model aslinya dari dalam pembungkus
                        base_model = final_estimator.regressor
                        self.model_type = type(base_model).__name__
                    else:
                        self.model_type = type(final_estimator).__name__
                else:
                    self.model_type = type(pipeline).__name__
            else:
                self.model_type = "Unknown_Model"
            self.r2 = r2_score(self.y_test_regression, self.y_pred_regression)
            self.mae = mean_absolute_error(self.y_test_regression, self.y_pred_regression)
            self.mape = mean_absolute_percentage_error(self.y_test_regression, self.y_pred_regression)
            self.smape = self.calculate_smape()
            self.rmse = np.sqrt(mean_squared_error(self.y_test_regression, self.y_pred_regression))

            self.metrics_regression = pd.DataFrame({
                'Model Type': [self.model_type],
                'R2': [self.r2],
                'MAE': [self.mae],
                'MAPE': [self.mape],
                'SMAPE': [self.smape],
                'RMSE': [self.rmse]
            })
            self.logger.info("[evaluate.py] Metrics calculated successfully.")

        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {e}")
            raise

        self.logger.info(f"R2 Score: {self.r2}")
        self.logger.info(f"MAE: {self.mae}")
        self.logger.info(f"MAPE: {self.mape}")
        self.logger.info(f"SMAPE: {self.smape}")
        self.logger.info(f"RMSE: {self.rmse}")
        
        self.logger.info(f"Creating Dataframe Model Diagnostics (10 biggest error)...")
        try:
            self.results_regression = pd.DataFrame({
                'item_id': self.df_meta_regression.loc[self.indices_test, 'item_id'],
                'title': self.df_meta_regression.loc[self.indices_test, 'title'],
                'is_boxset': self.df_meta_regression.loc[self.indices_test, 'is_boxset'],
                'is_special_edition': self.df_meta_regression.loc[self.indices_test, 'is_special_edition'],
                'volume_count': self.X_test_regression['volume_count'],
                'Actual': self.y_test_regression.values.ravel(),
                'Predicted': self.y_pred_regression.values.ravel()
            }, index=self.indices_test)
            self.logger.info(f"Dataframe diagnostics created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating dataframe diagnostics: {e}")
            raise
        
        self.results_regression['Residual'] = self.results_regression['Actual'] - self.results_regression['Predicted']
        self.results_regression['Abs_Error'] = np.abs(self.results_regression['Residual'])
        self.results_regression['APE (%)'] = (self.results_regression['Abs_Error'] / self.results_regression['Actual']) * 100 

        if(verbose == 1):
            self.logger.info(f"Displaying top 10 biggest error...")
            print(self.results_regression.sort_values(by='Abs_Error', ascending=False).head(10))
        
        self.logger.info(f"[evaluate.py] Evaluation succesfully completed.")
        self.output_json_regression()
        
if __name__ == "__main__":
    pipeline_evaluate = Pipeline_evaluate()
    pipeline_evaluate.evaluate_regression(verbose=1)