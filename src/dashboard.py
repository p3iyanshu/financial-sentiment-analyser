import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(
    page_title="Financial News Sentiment Analyser",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial News Sentiment → Market Impact Analyser")
st.markdown("Analysing how financial news sentiment correlates with stock price movements using **FinBERT** and **Random Forest**.")

@st.cache_data
def load_data():
    news_df = pd.read_csv("data/news_sentiment.csv")
    stock_df = pd.read_csv("data/stock_prices.csv")
    merged_df = pd.read_csv("data/merged_data.csv")
    return news_df, stock_df, merged_df

news_df, stock_df, merged_df = load_data()

st.sidebar.header("Filters")
companies = ["All"] + list(news_df["company"].unique())
selected_company = st.sidebar.selectbox("Select Company", companies)
selected_sentiment = st.sidebar.multiselect(
    "Filter by Sentiment",
    ["positive", "negative", "neutral"],
    default=["positive", "negative", "neutral"]
)

if selected_company != "All":
    filtered_news = news_df[
        (news_df["company"] == selected_company) &
        (news_df["sentiment"].isin(selected_sentiment))
    ]
    filtered_merged = merged_df[merged_df["company"] == selected_company]
else:
    filtered_news = news_df[news_df["sentiment"].isin(selected_sentiment)]
    filtered_merged = merged_df

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Headlines", len(filtered_news))
col2.metric("Positive", len(filtered_news[filtered_news["sentiment"] == "positive"]))
col3.metric("Negative", len(filtered_news[filtered_news["sentiment"] == "negative"]))
col4.metric("Neutral", len(filtered_news[filtered_news["sentiment"] == "neutral"]))

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_news["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
    fig_pie = px.pie(
        sentiment_counts, values="count", names="sentiment",
        color="sentiment", color_discrete_map=color_map
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("Sentiment Score vs Price Change %")
    fig_scatter = px.scatter(
        filtered_merged,
        x="avg_sentiment", y="price_change_pct",
        color="company", size="total_headlines",
        hover_data=["date", "company"],
        labels={
            "avg_sentiment": "Avg Sentiment Score",
            "price_change_pct": "Price Change %"
        }
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.subheader("Daily Sentiment Score Over Time")
fig_line = px.line(
    filtered_merged.sort_values("date"),
    x="date", y="avg_sentiment",
    color="company",
    markers=True,
    labels={"avg_sentiment": "Avg Sentiment Score", "date": "Date"}
)
fig_line.add_hline(y=0, line_dash="dot", line_color="gray")
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")
st.subheader("Correlation Analysis")
col_c, col_d = st.columns(2)

with col_c:
    corr_data = []
    for company in merged_df["company"].unique():
        df_c = merged_df[merged_df["company"] == company]
        corr = df_c["avg_sentiment"].corr(df_c["price_change_pct"])
        corr_data.append({"Company": company, "Correlation": round(corr, 4)})
    corr_df = pd.DataFrame(corr_data)
    fig_bar = px.bar(
        corr_df, x="Company", y="Correlation",
        color="Correlation", color_continuous_scale="RdYlGn",
        title="Sentiment-Price Correlation by Company"
    )
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_d:
    st.subheader("ML Model Charts")
    chart_options = {
        "Feature Importance": "data/charts/feature_importance.png",
        "SHAP Explainability": "data/charts/shap_importance.png",
        "Confusion Matrix": "data/charts/confusion_matrix.png",
        "Correlation Heatmap": "data/charts/correlation_heatmap.png"
    }
    selected_chart = st.selectbox("Select Chart", list(chart_options.keys()))
    chart_path = chart_options[selected_chart]
    if os.path.exists(chart_path):
        img = Image.open(chart_path)
        st.image(img, use_column_width=True)

st.markdown("---")
st.subheader("Raw Headlines with Sentiment")
st.dataframe(
    filtered_news[["date", "company", "headline", "sentiment", "confidence"]]
    .sort_values("date", ascending=False)
    .reset_index(drop=True),
    use_container_width=True
)

st.markdown("---")
st.markdown("Built with FinBERT + Random Forest | Data: NewsAPI + Yahoo Finance")