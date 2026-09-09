# %% [markdown]
# # 🎵 Spotify Song Popularity Prediction
# **Author:** Rishit Srivastava  
# **Task:** CodeNex AIML Club Recruitment — Task 2  
# **Objective:** Build a regression model to predict song popularity using audio features from the Spotify Tracks Dataset.
# 
# ---

# %% [markdown]
# ## 1. Setup & Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost - install if not available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    from xgboost import XGBRegressor
    HAS_XGB = True

print("All imports loaded successfully.")

# %% [markdown]
# ## 2. Load the Dataset

# %%
# Download from Kaggle
# If running on Colab, upload the CSV or use kaggle API
# For simplicity, we try reading from a direct path first

import os

# Option 1: If you uploaded the file to Colab
# from google.colab import files
# uploaded = files.upload()
