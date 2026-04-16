import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("data/charts", exist_ok=True)

def load_data():
    news_df = pd.read_csv("data/news_sentiment.csv")
    stock_df = pd.read_csv("data/stock_prices.csv")
    return news_df, stock_df


def aggregate_daily_sentiment(news_df):
    daily = news_df.groupby(["date", "company"]).agg(
        avg_sentiment=("sentiment_score", "mean"),
        total_headlines=("headline", "count"),
        positive=("sentiment", lambda x: (x == "positive").sum()),
        negative=("sentiment", lambda x: (x == "negative").sum()),
        neutral=("sentiment", lambda x: (x == "neutral").sum())
    ).reset_index()
    return daily


def merge_with_stocks(daily_sentiment, stock_df):
    company_ticker = {
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "Microsoft": "MSFT"
    }
    daily_sentiment["ticker"] = daily_sentiment["company"].map(company_ticker)
    merged = pd.merge(
        daily_sentiment,
        stock_df,
        on=["date", "ticker"],
        how="inner"
    )
    merged["price_change"] = merged["close"] - merged["open"]
    merged["price_change_pct"] = ((merged["close"] - merged["open"]) / merged["open"]) * 100
    return merged


def plot_sentiment_vs_price(merged_df):
    companies = merged_df["company"].unique()
    fig, axes = plt.subplots(len(companies), 2, figsize=(14, 5 * len(companies)))
    fig.suptitle("Sentiment Score vs Stock Price Change", fontsize=16, y=1.01)

    for i, company in enumerate(companies):
        df = merged_df[merged_df["company"] == company].copy()
        df = df.sort_values("date")

        ax1 = axes[i][0]
        ax1.bar(df["date"], df["avg_sentiment"],
                color=["green" if x > 0 else "red" if x < 0 else "gray"
                       for x in df["avg_sentiment"]])
        ax1.set_title(f"{company} — Daily Sentiment")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Avg Sentiment Score")
        ax1.tick_params(axis="x", rotation=45)

        ax2 = axes[i][1]
        ax2.bar(df["date"], df["price_change_pct"],
                color=["green" if x > 0 else "red"
                       for x in df["price_change_pct"]])
        ax2.set_title(f"{company} — Daily Price Change %")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price Change %")
        ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("data/charts/sentiment_vs_price.png", dpi=150, bbox_inches="tight")
    print("Saved: data/charts/sentiment_vs_price.png")
    plt.close()


def plot_correlation_heatmap(merged_df):
    cols = ["avg_sentiment", "total_headlines", "price_change_pct", "close"]
    corr = merged_df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap — Sentiment vs Market Data")
    plt.tight_layout()
    plt.savefig("data/charts/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved: data/charts/correlation_heatmap.png")
    plt.close()


def plot_sentiment_distribution(news_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    companies = news_df["company"].unique()

    for i, company in enumerate(companies):
        df = news_df[news_df["company"] == company]
        counts = df["sentiment"].value_counts()
        colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
        axes[i].pie(counts.values,
                    labels=counts.index,
                    colors=[colors[l] for l in counts.index],
                    autopct="%1.1f%%",
                    startangle=90)
        axes[i].set_title(f"{company} Sentiment Distribution")

    plt.suptitle("Sentiment Distribution by Company", fontsize=14)
    plt.tight_layout()
    plt.savefig("data/charts/sentiment_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved: data/charts/sentiment_distribution.png")
    plt.close()


def print_summary(merged_df):
    print("\n===== ANALYSIS SUMMARY =====")
    corr = merged_df["avg_sentiment"].corr(merged_df["price_change_pct"])
    print(f"Overall Sentiment-Price Correlation: {corr:.4f}")

    print("\nPer company correlation:")
    for company in merged_df["company"].unique():
        df = merged_df[merged_df["company"] == company]
        c = df["avg_sentiment"].corr(df["price_change_pct"])
        print(f"  {company}: {c:.4f}")

    print("\nAverage sentiment per company:")
    print(merged_df.groupby("company")["avg_sentiment"].mean().round(4))


def run_analysis():
    print("Loading data...")
    news_df, stock_df = load_data()

    print("Aggregating daily sentiment...")
    daily_sentiment = aggregate_daily_sentiment(news_df)

    print("Merging with stock prices...")
    merged_df = merge_with_stocks(daily_sentiment, stock_df)
    merged_df.to_csv("data/merged_data.csv", index=False)
    print(f"Merged dataset: {len(merged_df)} rows saved to data/merged_data.csv")

    print("\nGenerating charts...")
    plot_sentiment_vs_price(merged_df)
    plot_correlation_heatmap(merged_df)
    plot_sentiment_distribution(news_df)

    print_summary(merged_df)
    print("\nAll charts saved in data/charts/")


if __name__ == "__main__":
    run_analysis()