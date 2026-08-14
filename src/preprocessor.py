import sys
import os
from pathlib import Path

#DB Postgres Connection
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from config.config_script import load_dataframe

df_regression, df_classification = load_dataframe().fit()
print(df_regression.head(5))
        
        

