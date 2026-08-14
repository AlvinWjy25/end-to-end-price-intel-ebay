from pathlib import Path
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve()
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


