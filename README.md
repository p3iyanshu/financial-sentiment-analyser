# Financial News Sentiment → Market Impact Analyser

Analyses how financial news sentiment correlates with stock price movements using NLP and Machine Learning.

## What it does
- Scrapes financial news headlines using NewsAPI
- Scores sentiment using FinBERT (finance-specific BERT model)
- Fetches real stock price data using yfinance
- Performs correlation analysis between sentiment and price moves
- Trains a Random Forest classifier to predict price direction
- Explains predictions using SHAP values
- Visualises everything in an interactive Streamlit dashboard

## Key findings
- Tesla showed 0.72 sentiment-price correlation
- Apple showed 0.29 sentiment-price correlation
- Microsoft showed -0.67 (inverse) correlation
- Neutral news dominates (52%) vs positive (23%) and negative (24%)

## Tech stack
- Python, Pandas, NumPy
- FinBERT (HuggingFace Transformers)
- Scikit-learn (Random Forest)
- SHAP (model explainability)
- Plotly + Streamlit (dashboard)
- NewsAPI + yfinance (data sources)

## How to run
pip install -r requirements.txt
python src/scraper.py
python src/sentiment.py
python src/analysis.py
python src/ml_model.py
streamlit run src/dashboard.py

## Project structure
financial-sentiment-analyser/
├── data/              # CSV files and charts
├── src/
│   ├── scraper.py     # News + stock data collection
│   ├── sentiment.py   # FinBERT sentiment scoring
│   ├── analysis.py    # EDA + correlation analysis
│   ├── ml_model.py    # Random Forest + SHAP
│   └── dashboard.py   # Streamlit dashboard
└── README.md