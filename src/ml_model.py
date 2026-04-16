import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("data/charts", exist_ok=True)

def load_data():
    df = pd.read_csv("data/merged_data.csv")
    return df


def engineer_features(df):
    df = df.copy()
    df["price_went_up"] = (df["price_change_pct"] > 0).astype(int)
    df["sentiment_positive"] = (df["avg_sentiment"] > 0).astype(int)
    df["sentiment_negative"] = (df["avg_sentiment"] < 0).astype(int)
    df["high_volume_news"] = (df["total_headlines"] > df["total_headlines"].median()).astype(int)

    features = [
        "avg_sentiment",
        "total_headlines",
        "positive",
        "negative",
        "neutral",
        "sentiment_positive",
        "sentiment_negative",
        "high_volume_news"
    ]
    return df, features


def train_model(df, features):
    X = df[features]
    y = df["price_went_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== MODEL RESULTS =====")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Price Down", "Price Up"]))

    return model, X_train, X_test, y_test, y_pred, features


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Price Down", "Price Up"],
                yticklabels=["Price Down", "Price Up"])
    plt.title("Confusion Matrix — Random Forest")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("data/charts/confusion_matrix.png", dpi=150)
    print("Saved: data/charts/confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, features):
    importance = pd.Series(model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color="steelblue")
    plt.title("Feature Importance — What drives price prediction?")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("data/charts/feature_importance.png", dpi=150)
    print("Saved: data/charts/feature_importance.png")
    plt.close()


def plot_shap(model, X_train):
    print("\nGenerating SHAP explainability chart...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    plt.figure()
    shap.summary_plot(shap_vals, X_train,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("data/charts/shap_importance.png",
                dpi=150, bbox_inches="tight")
    print("Saved: data/charts/shap_importance.png")
    plt.close()


def run_ml():
    print("Loading merged data...")
    df = load_data()

    print("Engineering features...")
    df, features = engineer_features(df)

    print("Training Random Forest model...")
    model, X_train, X_test, y_test, y_pred, features = train_model(
        df, features
    )

    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, features)
    plot_shap(model, X_train)

    print("\nAll ML charts saved in data/charts/")
    print("\nProject Phase 5 Complete!")


if __name__ == "__main__":
    run_ml()