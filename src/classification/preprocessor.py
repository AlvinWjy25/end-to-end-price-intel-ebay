import sys
import os
import numpy as np
import pandas as pd
import mlflow
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings('ignore')

#DB Postgres Connection
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from config.config_script import (
    load_dataframe, setup_logger, 
    DF_TRAIN_CLASSIFICATION_PATH, DF_VAL_CLASSIFICATION_PATH, DF_TEST_CLASSIFICATION_PATH, 
    TITLE_TRAIN_PATH, TITLE_VAL_PATH, TITLE_TEST_PATH,
    DESCRIPTION_TRAIN_PATH, DESCRIPTION_VAL_PATH, DESCRIPTION_TEST_PATH,
    TITLE_VECTORIZER_PATH, DESCRIPTION_VECTORIZER_PATH, 
    ARTIFACT_DIR, PREPROCESSED_CLASSIFICATION_DIR
)

import random
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix

# NOTE: Why not on config_script?
# config_script is also used on regression, which could introduce overhead for regression pipeline
def set_seed(seed: int = 42, logger=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    if logger is not None:
        logger.info(f"Seed set to {seed}. cuDNN deterministic mode enabled.")

class preprocess_classification(load_dataframe):
    def __init__(self, random_state=42):
        super().__init__()
        self.logger = setup_logger('classification', 'preprocessor_run')
        self.random_state = random_state
        self.logger.info(f"ROOT DIR: {root_dir}")
        self.logger.info("Initialized preprocessing class.")

        self.df_raw: pd.DataFrame | None = None
        self.df_train: pd.DataFrame | None = None
        self.df_val: pd.DataFrame | None = None
        self.df_test: pd.DataFrame | None = None

    def set_seed(self, seed):
        try:
            set_seed(seed, self.logger)
        except Exception as e:
            self.logger.error(f'Error setting seed to {seed}!')
            self.logger.error(f'{e}')
            raise e

    def clean_data(self) -> pd.DataFrame:
        try:
            self.df_raw = self.df_classification
        except Exception as e:
            self.logger.error(f"Failed to load df_raw classification")
            self.logger.error(f"{e}")
            raise e
        
        if self.df_raw is None:
            raise RuntimeError("Panggil load_data() dulu sebelum clean_data().")
    
        before = len(self.df_raw)
    
        # Assume NULL or empty string/whitespace as empty
        title_empty = self.df_raw["title"].isna() | (self.df_raw["title"].str.strip() == "")
        desc_empty = self.df_raw["description"].isna() | (self.df_raw["description"].str.strip() == "")
    
        both_empty_mask = title_empty & desc_empty
        self.df_raw = self.df_raw.loc[~both_empty_mask].reset_index(drop=True)

        # Fill individual emptiness (title isn't empty but description is empty, and so on)
        # with empty string to prevent TF-IDF vectorizer crashing
        # while receiving None/NaN
        self.df_raw["title"] = self.df_raw["title"].fillna("")
        self.df_raw["description"] = self.df_raw["description"].fillna("")
    
        after = len(self.df_raw)
        self.logger.info(f"Dropped {before - after} row(s) with both title & description empty.")
        self.logger.info(f"Remaining: {after} rows.")
        return self.df_raw

    def stratified_split(
            self,
            val_size: float = 0.15,
            test_size: float = 0.15,
            random_state: int | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if self.df_raw is None:
            self.logger.error(f'Call load_data() dan clean_data() before stratified_split().')
            raise RuntimeError("Call load_data() dan clean_data() before stratified_split().")

        if random_state is None:
            random_state = self.random_state

        # Split df_raw into train/val/test with 70/15/15 proportion, stratified
        # based on `text_risk_score_v2` NOT `text_risk_score`
        
        # Step 1: divide test_size from entire data
        # df_raw (100%) -> df_train_val(85%) + df_test(15%)
        df_trainval, df_test = train_test_split(
            self.df_raw,
            test_size=test_size,
            stratify=self.df_raw["text_risk_score_v2"],
            random_state=random_state,
        )
        
        # Step 2: divide df_trainval, from val_size
        relative_val_size = val_size / (1 - test_size)
        df_train, df_val = train_test_split(
            df_trainval,
            test_size=relative_val_size,
            stratify=df_trainval["text_risk_score_v2"],
            random_state=random_state,
        )
        
        # reset_index 0..N for each split for easier tracking on PyTorch Dataset
        self.df_train = df_train.reset_index(drop=True)
        self.df_val = df_val.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)
        
        self._print_split_summary()
        return self.df_train, self.df_val, self.df_test

    def _print_split_summary(self) -> None:
        """Print size & proportion label for each split to verify"""
        for name, df in [("Train", self.df_train), ("Val", self.df_val), ("Test", self.df_test)]:
            risk_pct = (df["text_risk_score_v2"] == 70).mean() * 100
            self.logger.info(f"{name:5s}: {len(df):4d} rows | risk=70 proportion: {risk_pct:.2f}%")

    def fit_transform(self, SEED = 42):
        self.set_seed(SEED)
        self.df_classification = self.fit('classification')
        self.df_classification = self.clean_data()
        self.df_train, self.df_val, self.df_test = self.stratified_split()

        return self.df_train, self.df_val, self.df_test

class TextVectorizer:
    """Wrap independent title and description TF-IDF vectorizers."""

    def __init__(
        self,
        title_max_features: int = 2000,
        desc_max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ):
        """Configure separate vectorizers for title and description.

        The title vocabulary is usually smaller because titles are short.
        The description vocabulary can be larger because descriptions have
        more varied language. `min_df=2` removes one-off noise and typos.
        """
        self.title_vectorizer = TfidfVectorizer(
            max_features=title_max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            stop_words="english",
        )
        self.desc_vectorizer = TfidfVectorizer(
            max_features=desc_max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            stop_words="english",
        )
        self.is_fitted = False

    def fit(self, df_train) -> "TextVectorizer":
        """Fit both vectorizers on training data only."""
        self.title_vectorizer.fit(df_train["title"])
        self.desc_vectorizer.fit(df_train["description"])
        self.is_fitted = True

        print(f"Title vocabulary size : {len(self.title_vectorizer.vocabulary_)}")
        print(f"Description vocabulary size : {len(self.desc_vectorizer.vocabulary_)}")
        return self

    def transform(self, df) -> tuple[csr_matrix, csr_matrix]:
        """Transform title and description using the fitted vectorizers.

        Sparse matrices are returned because TF-IDF matrices are mostly zero.
        Conversion to dense tensors happens later, one batch at a time, in
        the PyTorch Dataset.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        title_matrix = self.title_vectorizer.transform(df["title"])
        desc_matrix = self.desc_vectorizer.transform(df["description"])
        return title_matrix, desc_matrix

    def save(self) -> None:

        """Save the fitted vectorizers for reuse during serving."""
        joblib.dump(self.title_vectorizer, f"{TITLE_VECTORIZER_PATH}")
        joblib.dump(self.desc_vectorizer, f"{DESCRIPTION_VECTORIZER_PATH}")
        print(f"Saved vectorizers can be found at: {ARTIFACT_DIR}")
    

    @classmethod
    def load(cls) -> "TextVectorizer":
        """Load previously fitted vectorizers without fitting again."""
        instance = cls()
        instance.title_vectorizer = joblib.load(f"{TITLE_VECTORIZER_PATH}")
        instance.desc_vectorizer = joblib.load(f"{DESCRIPTION_VECTORIZER_PATH}")
        instance.is_fitted = True
        return instance


def save_sparse_matrix_as_parquet(matrix: csr_matrix, path: Path) -> None:
    """Save a sparse TF-IDF matrix as a dense Parquet DataFrame."""
    pd.DataFrame(matrix.toarray()).to_parquet(path, index=False)
    

if __name__ == "__main__":
    logger = setup_logger('classification', 'preprocessor_run')
    logger.info("Running preprocessor.py directly...")

    try:
        logger.info("[ATTEMPTING]: Cleaning & Splitting the Dataframe...")
        df_train, df_val, df_test = preprocess_classification().fit_transform()
    except Exception as e:
        logger.error("[FAILED]: Fail to clean or splitting the Dataframe...")
        logger.error(f"{e}")
        raise e

    try:
        logger.info(f"[ATTEMPTING]: Saving splitted dataframe into {PREPROCESSED_CLASSIFICATION_DIR}")
        pd.DataFrame(df_train).to_parquet(DF_TRAIN_CLASSIFICATION_PATH, index=False)
        pd.DataFrame(df_val).to_parquet(DF_VAL_CLASSIFICATION_PATH, index=False)
        pd.DataFrame(df_test).to_parquet(DF_TEST_CLASSIFICATION_PATH, index=False)
        logger.info(f"[SUCCESS]: Splitted Dataframe succesfully saved!")
    except Exception as e:
        logger.error(f"[FAILED]: Error saving splitted dataframe!")
        logger.error(f"{e}")
        raise e

    try:
        logger.info("[ATTEMPTING]: Vectorizing title and description...")
        vectorizer = TextVectorizer(
            title_max_features=3000,
            desc_max_features=4000,
            ngram_range=(1, 2),
            min_df=2,
        )

        vectorizer.fit(df_train)
        
        title_train, desc_train = vectorizer.transform(df_train)
        title_val, desc_val = vectorizer.transform(df_val)
        title_test, desc_test = vectorizer.transform(df_test)

        logger.info("[SUCCESS]: Title & Description product succesfully vectorized!")
    except Exception as e:
        logger.error(f"[FAILED]: Error vectorizing title and description model!")
        logger.error(f"{e}")
        raise e

    try:
        logger.info("Saving title and description train/val/test..")
        save_sparse_matrix_as_parquet(title_train, TITLE_TRAIN_PATH)
        save_sparse_matrix_as_parquet(title_val, TITLE_VAL_PATH)
        save_sparse_matrix_as_parquet(title_test, TITLE_TEST_PATH)
        save_sparse_matrix_as_parquet(desc_train, DESCRIPTION_TRAIN_PATH)
        save_sparse_matrix_as_parquet(desc_val, DESCRIPTION_VAL_PATH)
        save_sparse_matrix_as_parquet(desc_test, DESCRIPTION_TEST_PATH)
        logger.info("Title and description matrices saved as Parquet.")
        
    except Exception as e:
        raise e

    try:
        logger.info(f"[ATTEMPTING]: Dumping title & description vectorizer into {ARTIFACT_DIR}...")
        vectorizer.save()
        logger.info(f"[SUCCESS]: Succesfully saved title & description vectorizer...")
    except Exception as e:
        logger.error(f"[FAILED]: Error saving title & description vectorizer caused by")
        logger.error(f'{e}')
        raise e

    logger.info(f"  Shapes:")
    logger.info(f"  title_train: {title_train.shape}, desc_train: {desc_train.shape}")
    logger.info(f"  title_val  : {title_val.shape}, desc_val  : {desc_val.shape}")
    logger.info(f"  title_test : {title_test.shape}, desc_test : {desc_test.shape}")
