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

plt.tight_layout()
plt.savefig('popularity_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPopularity stats: mean={df['popularity'].mean():.2f}, median={df['popularity'].median():.2f}, std={df['popularity'].std():.2f}")

# %% [markdown]
# ## 4. Preprocessing

# %% [markdown]
# ### 4.1 Handle Missing Values & Irrelevant Columns

# %%
print("=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0])
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Drop rows with missing values (if any)
df_clean = df.dropna().copy()
print(f"\nShape after dropping NaN: {df_clean.shape}")

# %%
# Drop irrelevant identifier columns
# track_id, artists, album_name, track_name are identifiers/text, not useful as numeric features
# We will NOT use popularity as an input feature (as per instructions)

columns_to_drop = ['track_id', 'artists', 'album_name', 'track_name', 'Unnamed: 0']
columns_to_drop = [c for c in columns_to_drop if c in df_clean.columns]

print(f"Dropping irrelevant columns: {columns_to_drop}")
df_clean = df_clean.drop(columns=columns_to_drop)

print(f"Remaining columns: {df_clean.columns.tolist()}")
print(f"Shape: {df_clean.shape}")

# %% [markdown]
# ### 4.2 Handle Categorical Variables

# %%
# Check for categorical columns
print("Data types:")
print(df_clean.dtypes)
print()

# 'explicit' is boolean - convert to int
if df_clean['explicit'].dtype == bool or df_clean['explicit'].dtype == object:
    df_clean['explicit'] = df_clean['explicit'].astype(int)
    print("Converted 'explicit' to integer (0/1)")

# 'track_genre' is categorical - encode it
print(f"\nUnique genres: {df_clean['track_genre'].nunique()}")
print(f"Sample genres: {df_clean['track_genre'].unique()[:10]}")

# Label encode the genre column
le_genre = LabelEncoder()
df_clean['track_genre_encoded'] = le_genre.fit_transform(df_clean['track_genre'])
df_clean = df_clean.drop(columns=['track_genre'])

print(f"\nGenre encoded. New shape: {df_clean.shape}")

# %%
# Final check on all columns
print("Final columns and types:")
print(df_clean.dtypes)
print(f"\nAny remaining non-numeric: {df_clean.select_dtypes(exclude=[np.number]).columns.tolist()}")

# %% [markdown]
# ### 4.3 Handle Duplicates

# %%
dupes = df_clean.duplicated().sum()
print(f"Duplicate rows: {dupes}")
if dupes > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"After removing duplicates: {df_clean.shape}")

# %% [markdown]
# ### 4.4 Correlation Analysis & Feature Selection

# %%
# Correlation heatmap
plt.figure(figsize=(14, 10))
corr_matrix = df_clean.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Correlation with popularity specifically
pop_corr = corr_matrix['popularity'].drop('popularity').sort_values(ascending=False)
print("Correlation with Popularity:")
print(pop_corr)

# Visualize it
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1DB954' if v > 0 else '#e74c3c' for v in pop_corr.values]
pop_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='black', alpha=0.8)
ax.set_title('Feature Correlation with Popularity', fontsize=14, fontweight='bold')
ax.set_xlabel('Pearson Correlation Coefficient')
ax.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_correlation_popularity.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.5 Feature Scaling

# %%
# Separate features and target
X = df_clean.drop(columns=['popularity'])
y = df_clean['popularity']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {X.columns.tolist()}")

# %%
# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# %%
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling applied (StandardScaler - zero mean, unit variance)")

# %% [markdown]
# ## 5. Model Training

# %% [markdown]
# I'm going to compare 4 different regression models:
# 1. **Ridge Regression** (linear baseline)
# 2. **Random Forest Regressor** (ensemble, bagging)
# 3. **Gradient Boosting Regressor** (ensemble, boosting)
# 4. **XGBoost Regressor** (optimized boosting)

# %%
# Define models
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
