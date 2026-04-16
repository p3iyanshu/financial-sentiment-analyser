# Financial News Sentiment Analyser

A data science project I built during my semester break to understand 
whether financial news sentiment has any relationship with stock price 
movements.

## The Idea

I always wondered if the tone of news around a company (positive, 
negative, neutral) has any effect on how its stock performs that day. 
This project is my attempt to actually test that using real data.

## What I Did

- Collected 30 days of news headlines for Apple, Tesla and Microsoft 
  using NewsAPI
- Used FinBERT (a BERT model trained specifically on financial text) 
  to classify each headline as positive, negative or neutral
- Pulled actual stock price data using yfinance
- Analysed the correlation between daily sentiment and price changes
- Trained a Random Forest model to predict if the stock price would 
  go up or down based on that day's news sentiment
- Added SHAP values to understand which features the model relied on
- Built a Streamlit dashboard to visualise everything interactively

## Results

| Company   | Sentiment-Price Correlation |
|-----------|-----------------------------|
| Tesla     | +0.72 (strong)              |
| Apple     | +0.29 (moderate)            |
| Microsoft | -0.67 (inverse)             |

Tesla showed the strongest signal which was surprising. Microsoft 
being negative was interesting — positive news about Microsoft often 
came on days prices dipped, possibly because good news was already 
priced in.

## What I Learned

- How to work with real APIs (NewsAPI, yfinance)
- How transformer models like BERT work for text classification
- Difference between correlation and causation (the model accuracy 
  is low because 30 days of data is not enough to generalise)
- How SHAP helps explain what a black-box model is actually doing
- End-to-end project structure — from raw data to deployed dashboard

## Honest Limitations

The ML model accuracy is not great — only 3 test samples because 
the dataset is small. This was my first time working with 
transformers so the pipeline is basic. Given more data (1-2 years) 
and more companies, the results would be more meaningful.

## Tech Used

Python · Pandas · HuggingFace Transformers · Scikit-learn · 
SHAP · Plotly · Streamlit · NewsAPI · yfinance

## How to Run

```bash
git clone https://github.com/p3iyanshu/financial-sentiment-analyser
cd financial-sentiment-analyser
pip install -r requirements.txt
```

Add your NewsAPI key in src/scraper.py at line 6, then run:

```bash
python src/scraper.py
python src/sentiment.py
python src/analysis.py
python src/ml_model.py
streamlit run src/dashboard.py
```

## Project Structure

```
financial-sentiment-analyser/
│
├── data/
│   ├── news_raw.csv
│   ├── news_sentiment.csv
│   ├── stock_prices.csv
│   ├── merged_data.csv
│   └── charts/
│
├── src/
│   ├── scraper.py
│   ├── sentiment.py
│   ├── analysis.py
│   ├── ml_model.py
│   └── dashboard.py
│
├── requirements.txt
└── README.md
```
