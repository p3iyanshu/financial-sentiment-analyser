from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

def load_finbert():
    print("Loading FinBERT model (first time may take 2-3 mins to download)...")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def get_sentiment(headline, tokenizer, model):
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["positive", "negative", "neutral"]
    scores = probs[0].tolist()

    sentiment = labels[scores.index(max(scores))]
    confidence = max(scores)
    return sentiment, round(confidence, 4)


def score_all_headlines():
    print("Reading news headlines...")
    news_df = pd.read_csv("data/news_raw.csv")

    tokenizer, model = load_finbert()

    sentiments = []
    confidences = []

    total = len(news_df)
    for i, headline in enumerate(news_df["headline"]):
        if i % 50 == 0:
            print(f"  Processing {i}/{total} headlines...")
        sentiment, confidence = get_sentiment(str(headline), tokenizer, model)
        sentiments.append(sentiment)
        confidences.append(confidence)

    news_df["sentiment"] = sentiments
    news_df["confidence"] = confidences

    score_map = {"positive": 1, "negative": -1, "neutral": 0}
    news_df["sentiment_score"] = news_df["sentiment"].map(score_map)

    news_df.to_csv("data/news_sentiment.csv", index=False)
    print(f"\nDone! Saved data/news_sentiment.csv")
    print(news_df["sentiment"].value_counts())


if __name__ == "__main__":
    score_all_headlines()