import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

NEWS_API_KEY = "f0568cf260e140f59ab65c1a60e67d82" 

COMPANIES = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT"
}

def fetch_news(company_name, from_date, to_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name + " stock",
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] != "ok":
        print(f"Error fetching news for {company_name}: {data.get('message')}")
        return []

    articles = []
    for article in data["articles"]:
        articles.append({
            "date": article["publishedAt"][:10],
            "company": company_name,
            "headline": article["title"]
        })
    return articles


def fetch_stock_prices(ticker, from_date, to_date):
    stock = yf.download(ticker, start=from_date, end=to_date, progress=False)
    stock = stock.reset_index()
    stock = stock[["Date", "Close", "Open"]]
    stock.columns = ["date", "close", "open"]
    stock["date"] = stock["date"].astype(str).str[:10]
    stock["ticker"] = ticker
    return stock


def run_pipeline():
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    all_news = []
    all_stocks = []

    for company, ticker in COMPANIES.items():
        print(f"Fetching news for {company}...")
        news = fetch_news(company, start_date, end_date)
        all_news.extend(news)

        print(f"Fetching stock prices for {ticker}...")
        stock = fetch_stock_prices(ticker, start_date, end_date)
        all_stocks.append(stock)

    news_df = pd.DataFrame(all_news)
    stock_df = pd.concat(all_stocks, ignore_index=True)

    news_df.to_csv("data/news_raw.csv", index=False)
    stock_df.to_csv("data/stock_prices.csv", index=False)

    print("\nDone! Files saved:")
    print(f"  data/news_raw.csv     — {len(news_df)} headlines")
    print(f"  data/stock_prices.csv — {len(stock_df)} price records")


if __name__ == "__main__":
    run_pipeline()