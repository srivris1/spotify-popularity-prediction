# Spotify Song Popularity Predictor

A machine learning project built for CodeNex AIML Task 2. This project uses regression models to predict a song's popularity (0-100) based strictly on its audio features and genre.

## Project Structure
- `spotify_popularity_prediction.py` -> The core script covering data preprocessing, model training, and evaluation.
- `Task_Report.txt` & `Task_Report.html` -> Final reports discussing methodology and results.
- `*.png` -> Visualizations generated during analysis (heatmaps, feature importance, etc.).
- `dataset.csv` -> Automatically fetched when running the script.

## Tech Stack
- **Python** (Pandas, NumPy)
- **Scikit-Learn** (Preprocessing, Baseline Models)
- **XGBoost** (Primary Regression Model)
- **Matplotlib & Seaborn** (Data Visualization)
- **Kagglehub** (Automated dataset retrieval)

## What I Learned
- **Feature Engineering:** Encoding highly cardinal features (like 114 unique genres) using Label Encoding avoids extreme sparsity while retaining predictive power for tree-based models.
- **Model Selection:** Tree-based ensemble methods (Random Forest, XGBoost) significantly outperformed standard linear models (Ridge Regression) when mapping complex audio attributes to popularity.
- **Data Insights:** High energy and specific genres strongly influence popularity, though subjective real-world metrics contain inherent noise.

## Difficulties Faced
The main challenge was the **high dimensionality of categorical features**, specifically the 114 unique genres. One-hot encoding them would drastically increase feature space and risk overfitting. I handled this by using `LabelEncoder`. While linear models struggle with this, tree-based regressors (like my chosen XGBoost) handle ordinal-encoded categories efficiently without assuming strict ordinality.

## How to Run
Simply execute `spotify_popularity_prediction.py`. The required Spotify dataset is fetched automatically via Kagglehub.
