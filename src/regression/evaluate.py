import pandas as pd
import pyarrow
import fastparquet
import numpy as np
import sys
import os
import joblib
from pathlib import Path
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from sklearn.model_selection import train_test_split
from config.config_script import load_dataframe, setup_logger, X_TRAIN_REGRESSION_PATH, X_TEST_REGRESSION_PATH, Y_TRAIN_REGRESSION_PATH, Y_TEST_REGRESSION_PATH, Y_PRED_REGRESSION_PATH, DF_META_REGRESSION_PATH, EVAL_SUMMARY_REGRESSION_PATH, EVAL_DATAFRAME_REGRESSION_PATH, MODEL_REGRESSION_PATH, EVAL_RANDOM_SEED_REGRESSION_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

class Pipeline_evaluate():
    def __init__(self):
        self.logger = setup_logger('regression', 'evaluate')
        self.logger.info("[evaluate.py] Reading test & prediction data...")
        try:
            self.X_train_regression = pd.read_parquet(X_TRAIN_REGRESSION_PATH)
            self.X_test_regression = pd.read_parquet(X_TEST_REGRESSION_PATH)
            self.y_train_regression = pd.read_parquet(Y_TRAIN_REGRESSION_PATH)
            self.y_test_regression = pd.read_parquet(Y_TEST_REGRESSION_PATH)
            self.y_pred_regression = pd.read_parquet(Y_PRED_REGRESSION_PATH)
            self.df_meta_regression = pd.read_parquet(DF_META_REGRESSION_PATH)
            self.indices_train = pd.read_parquet(X_TRAIN_REGRESSION_PATH).index
            self.indices_test = pd.read_parquet(X_TEST_REGRESSION_PATH).index
            self.logger.info("[evaluate.py] Data loaded successfully.")
        except Exception as e:
            self.logger.error(f"[evaluate.py] Error loading data: {e}, make sure you run preprocessor.py & train.py first!")
            raise
        
    def calculate_smape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        epsilon = 1e-10
        denominator = np.where(denominator == 0, epsilon, denominator)
        return np.mean(np.abs(y_pred - y_true) / denominator) * 100

    def evaluate_random_split(self, grid_search):
        self.logger.info("Starting Evaluation on different seeds ([1, 2, 3, 4, 5, 42, 100])...")
        self.X_full = pd.concat([self.X_train_regression, self.X_test_regression], axis=0)
        self.y_full = pd.concat([self.y_train_regression, self.y_test_regression]).squeeze()
        self.idx_full = self.X_full.index

        self.seeds = [1, 2, 3, 4, 5, 42, 100]
        self.r2_list, self.mae_list, self.rmse_list, self.mape_list, self.smape_list = [], [], [], [], []

        best_params_only = {
            k.replace('regressor__regressor__', ''): v 
            for k, v in grid_search.best_params_.items()
        }

        price_bins = pd.qcut(self.y_full, q=4, labels=False, duplicates='drop')

        for seed in self.seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(
               self.X_full, self.y_full, test_size=0.2, random_state=seed, stratify=price_bins
            )

            from sklearn.base import clone
            model_clone = clone(grid_search.best_estimator_)
            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict(X_te)

            r2 = r2_score(y_te, y_pred)
            mae = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            mape = mean_absolute_percentage_error(y_te, y_pred)
            smape = self.calculate_smape(y_te, y_pred)

            self.r2_list.append(r2)
            self.mae_list.append(mae)
            self.rmse_list.append(rmse)
            self.mape_list.append(mape)
            self.smape_list.append(smape)
            # print(f"Seed {seed:3d} | R2 = {r2:.3f} | MAE = {mae:.2f}")

        def summarize_metric(values, name):
            values = np.array(values)
            return {
                'Metric': name,
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Range': values.max() - values.min()
            }

        summary_df = pd.DataFrame([
            summarize_metric(self.r2_list, 'R2'),
            summarize_metric(self.mae_list, 'MAE'),
            summarize_metric(self.rmse_list, 'RMSE'),
            summarize_metric(self.mape_list, 'MAPE'),
            summarize_metric(self.smape_list, 'SMAPE'),
        ])

        summary_df = summary_df.round(3)
        self.logger.info(f"Summary for each split:\n{summary_df}")

        return summary_df

    def output_json_regression(self):
        self.logger.info(f"Saving Model Evaluation Summary...")
        try:
            self.metrics_regression.to_json(EVAL_SUMMARY_REGRESSION_PATH, orient='records', indent=4)
            self.logger.info(f"Evaluation metrics saved to {EVAL_SUMMARY_REGRESSION_PATH}")

            try:
                self.summary_df.to_json(EVAL_RANDOM_SEED_REGRESSION_PATH, orient = 'records', indent = 4)
                self.logger.info(f"Evaluation metrics saved to {EVAL_RANDOM_SEED_REGRESSION_PATH}")
            except Exception as e:
                self.logger.info(f"[SKIP RANDOM SEED evaluation]: To save evaluation_random_seed.json, you must run pipeline.py or train.py directly!")

            self.logger.info(f"Printing evaluation report into JSON format...")
            self.results_regression.to_json(EVAL_DATAFRAME_REGRESSION_PATH, orient='records', indent=4)
            self.logger.info(f"Evaluation dataframe saved to {EVAL_DATAFRAME_REGRESSION_PATH}")
        except Exception as e:
            self.logger.error(f"Error saving JSON output: {e}")
            raise
        
    def evaluate_regression(self, verbose, cv_results = None, best_index = None, pipeline = None, grid_search = None):
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
                self.model_type = "unknown_model"

            self.r2 = r2_score(self.y_test_regression, self.y_pred_regression)
            self.mae = mean_absolute_error(self.y_test_regression, self.y_pred_regression)
            self.mape = mean_absolute_percentage_error(self.y_test_regression, self.y_pred_regression)
            self.smape = self.calculate_smape(
                self.y_test_regression,
                self.y_pred_regression
            )
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
        self.logger.info("=" * 75)

        if(best_index != None and cv_results != None):
            self.logger.info(f"Evaluating Tuned Pipeline Regression by fold:")
            fold_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(15)]

            self.logger.info(f"Mean R^2   : {np.mean(fold_scores):.3f}")
            self.logger.info(f"Std R^2    : {np.std(fold_scores):.3f}")
            self.logger.info(f"Min R^2    : {np.min(fold_scores):.3f}")
            self.logger.info(f"Max R^2    : {np.max(fold_scores):.3f}")

            self.logger.info("=" * 75)

        if(grid_search != None):
            self.logger.info(f"Evaluating Tuned Pipeline Regression random seed split:")
            try:
                self.summary_df = self.evaluate_random_split(grid_search)
            except Exception as e:
                self.logger.info(f"Fail evaluating tuned pipeline with random seed split!")
                self.logger.error(f"{e}")
                raise e
        else:
            self.logger.info(f"Tuned Pipeline random split evaluation can only be triggered by running pipeline.py")
        
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
            self.logger.info(f"{self.results_regression.sort_values(by='Abs_Error', ascending=False).head(10)}")
        
        self.logger.info(f"[evaluate.py] Evaluation succesfully completed.")
        self.output_json_regression()
        
if __name__ == "__main__":
    pipeline_evaluate = Pipeline_evaluate()
    pipeline_evaluate.evaluate_regression(verbose=1, cv_results=None, best_index=None, pipeline = None)