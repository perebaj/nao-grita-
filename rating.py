import json

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Example data
data = [
    {
        "rate": 10.0,
        "sentiment_score": 0.9,
    },
    {
        "rate": 7.5,
        "sentiment_score": 0.5,
    },
    {
        "rate": 5.0,
        "sentiment_score": 0.0,
    },
    {
        "rate": 3.0,
        "sentiment_score": -0.3,
    },
    {
        "rate": 1.0,
        "sentiment_score": -0.8,
    },
]


def calculate_rating(data: dict) -> float:
    # Extract rate and sentiment_score
    rates = np.array([item["rate"] for item in data]).reshape(-1, 1)
    sentiment_scores = np.array([item["sentiment_score"] for item in data]).reshape(-1, 1)

    # Normalize both features to 0-1 range
    scaler_rate = MinMaxScaler(feature_range=(0, 1))
    scaler_sentiment = MinMaxScaler(feature_range=(0, 1))

    normalized_rate = scaler_rate.fit_transform(rates)
    normalized_sentiment = scaler_sentiment.fit_transform(sentiment_scores)

    # Assign weights
    rate_weight = 0.3
    sentiment_weight = 0.7

    # Calculate final score
    final_scores = (rate_weight * normalized_rate) + (sentiment_weight * normalized_sentiment)

    # Attach final scores back to the data
    for i, item in enumerate(data):
        item["final_rating"] = float(final_scores[i][0])

    # save the enriched influencers
    with open("data/influencers_enriched3.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

with open("data/influencers_enriched2.json", "r", encoding="utf-8") as file:
    data = json.load(file)

calculate_rating(data)
