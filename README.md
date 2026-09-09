# Spotify Popularity Predictor

Hey! This is my project for the CodeNex AIML Task 2. 
I grabbed the Spotify Tracks Dataset from Kaggle to see if I could predict how popular a song will be (on a scale from 0 to 100) just based on its audio features. 

## Project structure

- `spotify_popularity_prediction.py` -> the main python script where all the magic happens.
- `Task_Report.txt` & `Task_Report.html` -> my final report discussing my findings.
- `*.png` -> various charts and graphs generated during the analysis (like correlation heatmaps, feature importance, etc.).
- `dataset.csv` -> the dataset (downloads automatically if you run the script).

## What I analyzed

I basically wanted to see what makes a song popular. I looked into:
- **Audio features:** Things like how danceable a song is, its energy level, loudness, acousticness, etc.
- **Genres:** The dataset has around 114 different genres. I had to convert these into numbers so the model could understand them.

I checked how these features correlate with a song's popularity. Turns out, things like genre and energy matter a lot, but predicting popularity perfectly is really hard because it depends on a lot of outside factors (like marketing, artist fame, or TikTok trends) that aren't in the audio.

## Techs used

- **Python** (Pandas, NumPy for data manipulation)
- **Scikit-learn** (for data preprocessing and building baseline models)
- **Matplotlib & Seaborn** (for data visualization)
- **XGBoost** (the most accurate model I tried!)
- **Kagglehub** (to easily download the dataset directly in the script)

## My Approach

1. **Cleaning:** Dropped useless text columns like track ID and artist names. Cleaned up missing values and duplicates.
2. **Encoding:** Used `LabelEncoder` to change categorical stuff like genres into numbers.
3. **Training:** I tested 4 different models to see what works best:
   - Ridge Regression (basic, didn't do so well)
   - Random Forest
   - Gradient Boosting
   - XGBoost (the clear winner 🏆)
4. **Results:** XGBoost gave the best results. It's not 100% accurate (because popularity is subjective and trend-driven), but it learned the patterns in the data way better than the others.

## How to run

Just run `spotify_popularity_prediction.py`. It uses `kagglehub` to fetch the dataset automatically, so you don't even have to download it yourself.

Hope you like it!
