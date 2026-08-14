from pathlib import Path
import pandas as pd
import os

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs" / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print(f'ROOT DIR: {ROOT_DIR}\n')
ARTIFACT_DIR = Path(ROOT_DIR / 'src' / 'artifacts')
ARTIFACT_DIR.mkdir(parents = True, exist_ok = True)

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
        self.create_connection()
        df_regression = self.load_regression_data()
        df_classification = self.load_classification_data()
        return df_regression, df_classification


