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

# Option 2: Direct download using kagglehub (Colab has this pre-installed usually)
try:
    import kagglehub
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    csv_path = os.path.join(path, "dataset.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded via kagglehub from: {csv_path}")
except Exception:
    # Fallback: try loading from current directory
    try:
        df = pd.read_csv("dataset.csv")
        print("Loaded from local file.")
    except FileNotFoundError:
        print("Please upload dataset.csv to the working directory or install kagglehub.")
        raise

print(f"\nDataset shape: {df.shape}")
df.head()

# %% [markdown]
# ## 3. Initial Exploration

# %%
print("=== Dataset Info ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:")
df.describe()

# %%
# Check the columns we have
print("Columns:", df.columns.tolist())

# %%
# Distribution of the target variable - popularity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['popularity'], bins=50, color='#1DB954', edgecolor='black', alpha=0.8)
axes[0].set_title('Distribution of Popularity', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Popularity')
axes[0].set_ylabel('Count')
axes[0].axvline(df['popularity'].mean(), color='red', linestyle='--', label=f"Mean: {df['popularity'].mean():.1f}")
axes[0].legend()

axes[1].boxplot(df['popularity'], vert=True, patch_artist=True, 
                boxprops=dict(facecolor='#1DB954', alpha=0.7))
axes[1].set_title('Popularity Boxplot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Popularity')
