from pathlib import Path
import pandas as pd
import pyarrow
import fastparquet
import os

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs" / "pipeline_logs"
DBT_LOG_DIR = ROOT_DIR / "logs" / "dbt_logs"
DBT_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# print(f'ROOT DIR: {ROOT_DIR}\n')
ARTIFACT_DIR = Path(ROOT_DIR / 'src' / 'artifacts')
ARTIFACT_DIR.mkdir(parents = True, exist_ok = True)

PREPROCESSED_DIR = Path(ARTIFACT_DIR / 'preprocessed' / 'regression')
PREPROCESSED_DIR.mkdir(parents = True, exist_ok = True)

PREPROCESSED_CLASSIFICATION_DIR = Path(ARTIFACT_DIR / 'preprocessed' / 'classification')
PREPROCESSED_CLASSIFICATION_DIR.mkdir(parents = True, exist_ok = True)

X_TRAIN_PATH = PREPROCESSED_DIR / 'X_train.parquet'
X_TEST_PATH = PREPROCESSED_DIR / 'X_test.parquet'
Y_TRAIN_PATH = PREPROCESSED_DIR / 'y_train.parquet'
Y_TEST_PATH = PREPROCESSED_DIR / 'y_test.parquet'
Y_PRED_PATH = PREPROCESSED_DIR / 'y_pred.parquet'
DF_META_PATH = PREPROCESSED_DIR / 'df_meta.parquet'

EVALUATION_DIR = Path(ARTIFACT_DIR / 'evaluation')
EVALUATION_DIR.mkdir(parents = True, exist_ok = True)
(EVALUATION_DIR / 'regression').mkdir(parents = True, exist_ok = True)
(EVALUATION_DIR / 'classification').mkdir(parents = True, exist_ok = True)

EVAL_SUMMARY_REGRESSION_PATH = EVALUATION_DIR / 'regression' / 'evaluation_report.json'
EVAL_DATAFRAME_REGRESSION_PATH = EVALUATION_DIR / 'regression' / 'evaluation_dataframe.json'
EVAL_RANDOM_SEED_REGRESSION_PATH = EVALUATION_DIR / 'regression' / 'evaluation_random_seed.json'

EVAL_SUMMARY_CLASSIFICATION_PATH = EVALUATION_DIR / 'classification' / 'evaluation_report.json'
EVAL_DATAFRAME_CLASSIFICATION_PATH = EVALUATION_DIR / 'classification' / 'evaluation_dataframe.json'

ARTIFACT_MODEL_PATH = Path(ARTIFACT_DIR / 'models')
ARTIFACT_MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_REGRESSION_PATH = Path(ARTIFACT_DIR / 'models' / 'final_regression.joblib')
MODEL_CLASSIFICATION_PATH = Path(ARTIFACT_DIR / 'models' / 'final_classification.joblib')


def setup_logger(run_type: str) -> logging.Logger:
    """
    Configures a logger to output to both console and a timestamped file.
    Resolves directories dynamically to prevent path issues.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = LOG_DIR / f"{timestamp}_{run_type}.log"

    logger = logging.getLogger(run_type)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console Handler (Cleaner terminal output)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s]: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger

numeric_features = [
    'title_length', 
    'title_word_count', 
    'volume_count',
    'text_risk_score',
    'volume_tier_encoded',
    'condition_encoded',
    'total_bonus_count'
]
        
categorical_features = [
    'currency', 
    'seller_location_grouped'
]
        
boolean_features = [
    'is_boxset', 
    'is_special_edition', 
    'boxset_side_story_edition_included', 
    'standalone_side_story_edition',
    'is_first_print',
    'has_signature',
    'has_merch',
    'has_paper_extra',
]
        
feature_cols = numeric_features + categorical_features + boolean_features
target_col = 'price'

class load_dataframe:
    def __init__(self):
        self.DB_USER = os.getenv("DB_USER", "alvin")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "devpassword")
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = os.getenv("DB_PORT", "5432")
        self.DB_NAME = os.getenv("DB_NAME", "price_intelligence")

    def create_connection(self):
        self.QUERY_CREATE_ENGINE = (f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}")
        self.engine = create_engine(self.QUERY_CREATE_ENGINE)
        return self.engine
    
    def load_regression_data(self):
        QUERY_REGRESSION = "SELECT * FROM public.fct_ebay_listings"
        self.query_regression = QUERY_REGRESSION
        self.df_regression = pd.read_sql(self.query_regression, self.engine)
        return self.df_regression
    
    def load_classification_data(self):
        QUERY_CLASSIFICATION = "SELECT * FROM public.fct_ebay_listings"

        self.query_classification = QUERY_CLASSIFICATION
        self.df_classification = pd.read_sql(self.query_classification, self.engine)
        return self.df_classification
    
    def fit(self):
        try:
            self.create_connection()
            df_regression = self.load_regression_data()
            df_classification = self.load_classification_data()
        except Exception as e:
            print("Unable to connect to database, make sure it's Online on Docker!")
            raise e
            
        return df_regression, df_classification


