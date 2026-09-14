

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

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    from xgboost import XGBRegressor
    HAS_XGB = True

print("All imports loaded successfully.")



import os


try:
    import kagglehub
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    csv_path = os.path.join(path, "dataset.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded via kagglehub from: {csv_path}")
except Exception:
    try:
        df = pd.read_csv("dataset.csv")
        print("Loaded from local file.")
    except FileNotFoundError:
        print("Please upload dataset.csv to the working directory or install kagglehub.")
        raise

print(f"\nDataset shape: {df.shape}")
df.head()


print("=== Dataset Info ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:")
df.describe()

print("Columns:", df.columns.tolist())

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



print("=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0])
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

df_clean = df.dropna().copy()
print(f"\nShape after dropping NaN: {df_clean.shape}")


columns_to_drop = ['track_id', 'artists', 'album_name', 'track_name', 'Unnamed: 0']
columns_to_drop = [c for c in columns_to_drop if c in df_clean.columns]

print(f"Dropping irrelevant columns: {columns_to_drop}")
df_clean = df_clean.drop(columns=columns_to_drop)

print(f"Remaining columns: {df_clean.columns.tolist()}")
print(f"Shape: {df_clean.shape}")


print("Data types:")
print(df_clean.dtypes)
print()

if df_clean['explicit'].dtype == bool or df_clean['explicit'].dtype == object:
    df_clean['explicit'] = df_clean['explicit'].astype(int)
    print("Converted 'explicit' to integer (0/1)")

print(f"\nUnique genres: {df_clean['track_genre'].nunique()}")
print(f"Sample genres: {df_clean['track_genre'].unique()[:10]}")

le_genre = LabelEncoder()
df_clean['track_genre_encoded'] = le_genre.fit_transform(df_clean['track_genre'])
df_clean = df_clean.drop(columns=['track_genre'])

print(f"\nGenre encoded. New shape: {df_clean.shape}")

print("Final columns and types:")
print(df_clean.dtypes)
print(f"\nAny remaining non-numeric: {df_clean.select_dtypes(exclude=[np.number]).columns.tolist()}")


dupes = df_clean.duplicated().sum()
print(f"Duplicate rows: {dupes}")
if dupes > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"After removing duplicates: {df_clean.shape}")


plt.figure(figsize=(14, 10))
corr_matrix = df_clean.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

pop_corr = corr_matrix['popularity'].drop('popularity').sort_values(ascending=False)
print("Correlation with Popularity:")
print(pop_corr)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1DB954' if v > 0 else '#e74c3c' for v in pop_corr.values]
pop_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='black', alpha=0.8)
ax.set_title('Feature Correlation with Popularity', fontsize=14, fontweight='bold')
ax.set_xlabel('Pearson Correlation Coefficient')
ax.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_correlation_popularity.png', dpi=150, bbox_inches='tight')
plt.show()


X = df_clean.drop(columns=['popularity'])
y = df_clean['popularity']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling applied (StandardScaler - zero mean, unit variance)")



models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)
    
    if name == 'Ridge Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")


comparison_df = pd.DataFrame({
    name: {
        'MSE': res['MSE'],
        'RMSE': res['RMSE'],
        'MAE': res['MAE'],
        'R² Score': res['R2']
    } for name, res in results.items()
}).T

comparison_df = comparison_df.sort_values('R² Score', ascending=False)
print("\n=== MODEL COMPARISON ===")
print(comparison_df.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['RMSE', 'MAE', 'R² Score']
colors_list = ['#1DB954', '#3498db', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    bars = axes[i].bar(comparison_df.index, comparison_df[metric], 
                       color=colors_list[:len(comparison_df)], edgecolor='black', alpha=0.85)
    axes[i].set_title(metric, fontsize=14, fontweight='bold')
    axes[i].set_xticklabels(comparison_df.index, rotation=30, ha='right', fontsize=9)
    axes[i].set_ylabel(metric)
    
    for bar, val in zip(bars, comparison_df[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01*bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

best_model_name = comparison_df['R² Score'].idxmax()
best_model_results = results[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {best_model_results['R2']:.4f}")
print(f"   RMSE: {best_model_results['RMSE']:.4f}")
print(f"   MAE: {best_model_results['MAE']:.4f}")


y_pred_best = best_model_results['predictions']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_test, y_pred_best, alpha=0.15, s=10, color='#1DB954', edgecolors='none')
axes[0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Popularity', fontsize=12)
axes[0].set_ylabel('Predicted Popularity', fontsize=12)
axes[0].set_title(f'{best_model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_xlim(-5, 105)
axes[0].set_ylim(-5, 105)

residuals = y_test - y_pred_best
axes[1].hist(residuals, bins=60, color='#3498db', edgecolor='black', alpha=0.8, density=True)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1].text(0.05, 0.95, f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}', 
             transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()


if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    importances = best_model_results['model'].feature_importances_
else:
    importances = results['Random Forest']['model'].feature_importances_

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
feat_imp.plot(kind='barh', ax=ax, color='#1DB954', edgecolor='black', alpha=0.85)
ax.set_title('Feature Importance (Best Model)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 5 Most Important Features:")
for i, (feat, imp) in enumerate(feat_imp.sort_values(ascending=False).head(5).items()):
    print(f"  {i+1}. {feat}: {imp:.4f}")


error_analysis = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_best,
    'abs_error': np.abs(y_test.values - y_pred_best),
    'residual': y_test.values - y_pred_best
})

print("=== Hardest to Predict (Largest Absolute Errors) ===")
hardest = error_analysis.nlargest(10, 'abs_error')
print(hardest.to_string(index=False))

error_analysis['pop_bin'] = pd.cut(error_analysis['actual'], 
                                    bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                    labels=['0-10', '10-20', '20-30', '30-40', '40-50', 
                                           '50-60', '60-70', '70-80', '80-90', '90-100'])

error_by_bin = error_analysis.groupby('pop_bin')['abs_error'].agg(['mean', 'std', 'count'])
print("\n=== Mean Absolute Error by Popularity Range ===")
print(error_by_bin.to_string())

fig, ax = plt.subplots(figsize=(12, 5))
error_by_bin['mean'].plot(kind='bar', ax=ax, color='#e74c3c', edgecolor='black', alpha=0.8, yerr=error_by_bin['std'])
ax.set_title('Mean Absolute Error by Popularity Range', fontsize=14, fontweight='bold')
ax.set_xlabel('Popularity Range')
ax.set_ylabel('Mean Absolute Error')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('error_by_popularity.png', dpi=150, bbox_inches='tight')
plt.show()


thresholds = [5, 10, 15, 20, 25]
print("=== Prediction Accuracy within Thresholds ===")
for t in thresholds:
    pct = (error_analysis['abs_error'] <= t).mean() * 100
    print(f"  Within ±{t} popularity points: {pct:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ranges = [(0, 25, 'Low (0-25)'), (25, 50, 'Medium-Low (25-50)'), 
          (50, 75, 'Medium-High (50-75)'), (75, 100, 'High (75-100)')]

for ax, (lo, hi, label) in zip(axes.flatten(), ranges):
    mask = (error_analysis['actual'] >= lo) & (error_analysis['actual'] <= hi)
    subset = error_analysis[mask]
    ax.scatter(subset['actual'], subset['predicted'], alpha=0.2, s=8, color='#1DB954')
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
    ax.set_title(f'{label}\nMAE: {subset["abs_error"].mean():.2f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_xlim(lo-5, hi+5)

plt.suptitle('Predictions by Popularity Range', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('predictions_by_range.png', dpi=150, bbox_inches='tight')
plt.show()


if best_model_name == 'Ridge Regression':
    cv_scores = cross_val_score(best_model_results['model'], X_train_scaled, y_train, 
                                 cv=5, scoring='r2', n_jobs=-1)
else:
    cv_scores = cross_val_score(best_model_results['model'], X_train, y_train, 
                                 cv=5, scoring='r2', n_jobs=-1)

print(f"=== 5-Fold Cross-Validation ({best_model_name}) ===")
print(f"R² scores: {cv_scores}")
print(f"Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


print("=" * 60)
print("         SPOTIFY POPULARITY PREDICTION - SUMMARY")
print("=" * 60)
print()
print(f"Dataset: {df.shape[0]} tracks, {df.shape[1]} original features")
print(f"After preprocessing: {df_clean.shape[0]} rows, {X.shape[1]} features")
print()
print("Models Compared:")
for name in comparison_df.index:
    r2 = results[name]['R2']
    rmse = results[name]['RMSE']
    print(f"  - {name}: R²={r2:.4f}, RMSE={rmse:.4f}")
print()
print(f"🏆 Best Model: {best_model_name}")
print(f"   R² Score: {best_model_results['R2']:.4f}")
print(f"   RMSE: {best_model_results['RMSE']:.4f}")
print(f"   MAE: {best_model_results['MAE']:.4f}")
print()
print("Top Features:")
for i, (feat, imp) in enumerate(feat_imp.sort_values(ascending=False).head(5).items()):
    print(f"  {i+1}. {feat}")
print()
print("Key Insights:")
print("  - Songs with very low popularity (0-10) are the hardest to predict")
print("  - Genre encoding plays a major role in prediction")
print("  - Audio features like energy, loudness, and danceability are significant")
print("  - Tree-based models significantly outperform linear regression")
print()
print("Future Improvements:")
print("  - Try neural network based approaches")
print("  - Use artist popularity as a feature (external data)")
print("  - Experiment with more hyperparameter tuning (GridSearch/Bayesian)")
print("  - Apply target transformation (log) for skewed popularity distribution")
print("=" * 60)

print("\nSaved plots:")
print("  1. popularity_distribution.png")
print("  2. correlation_heatmap.png")
print("  3. feature_correlation_popularity.png")
print("  4. model_comparison.png")
print("  5. actual_vs_predicted.png")
print("  6. feature_importance.png")
print("  7. error_by_popularity.png")
print("  8. predictions_by_range.png")
