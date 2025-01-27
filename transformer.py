import json

import pandas as pd
import torch
from thefuzz import fuzz
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the tokenizer and model
# Model reference: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
MODEL_NAME = "cardiffnlp/xlm-roberta-base-tweet-sentiment-pt"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def sentiment_analysis(text: str) -> dict:
    """
    Perform sentiment analysis on the given text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing the sentiment and sentiment score.

    Example:
        >>> sentiment_analysis("I love this product!")
        {'sentiment': 'positive', 'sentiment_score': 0.98765}
    """

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Get the predicted sentiment
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Model-specific class labels (check the model's documentation)
    class_labels = ["negative", "neutral", "positive"]

    # Output sentiment and probabilities
    sentiment = class_labels[predicted_class]
    sentiment_score = probabilities[0, predicted_class].item()

    return {"sentiment": sentiment, "sentiment_score": round(sentiment_score, 5)}  # 5 decimal places


def call_sentiment_analysis():
    with open("data/influencers_parsed_final.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    influencers = []
    for influencer in data:
        sentiment = sentiment_analysis(influencer["work_description"] + " " + influencer["advice_description"])
        influencer = {**influencer, **sentiment}
        influencers.append(influencer)

    # save the results to a file
    with open("data/influencers_enriched_robertasentiment_ptbr.json", "w", encoding="utf-8") as file:
        json.dump(influencers, file, ensure_ascii=False, indent=4)


def calculate_final_rating(data: dict):
    """
    Calculates the final rating based on the given data.

    Args:
        data (dict): A dictionary containing the following keys:
            - rate (float): The rating value.
            - sentiment_score (float): The sentiment score.
            - sentiment (str): The sentiment category (positive, neutral, negative).

    Returns:
        None

    Modifies:
        Adds the 'final_rating' key to the 'data' dictionary with the calculated final rating value.

    """
    w_rate = 0.6
    w_sentiment = 0.4

    # normalize rate
    rate_normalize = data["rate"] / 10.0
    sentiment_multiplier = {"positive": 1.0, "neutral": 0.5, "negative": 0.2}

    adjusted_sentiment_score = data["sentiment_score"] * sentiment_multiplier[data["sentiment"]]

    final_rating = w_rate * rate_normalize + w_sentiment * adjusted_sentiment_score

    data["final_rating"] = round(final_rating, 5)


def call_calculate_final_rating():
    with open("data/influencers_enriched_robertasentiment_ptbr.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    for influencer in data:
        calculate_final_rating(influencer)

    # save the results to a file
    with open("data/influencers_enriched_robertasentiment_ptbr2.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def rank():
    """
    Ranks influencers based on their final rating, sentiment score, and rate.

    Reads data from a JSON file, creates a DataFrame, and sorts it based on the specified columns.
    Adds a ranking column to the DataFrame and displays the top 30 ranked influencers.

    Parameters:
        None

    Returns:
        None
    """
    with open("data/influencers_enriched_robertasentiment_ptbr2.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Sort by final_rating (descending), sentiment_score (descending), and rate (descending)
    df = df.sort_values(by=["final_rating", "sentiment_score", "rate"], ascending=[False, False, False]).reset_index(
        drop=True
    )

    # Add a ranking column
    df["rank"] = df.index + 1

    # Display the ranked DataFrame
    print(df[["rank", "id", "nickname", "final_rating", "sentiment_score", "rate"]][:30])


def fuzz_analysis():
    """
    Perform fuzzy matching analysis on influencers' data.

    This function reads data from a JSON file, performs fuzzy matching on the influencers' nicknames and names,
    and saves the matched pairs to another JSON file.

    Returns:
        None
    """
    with open("data/influencers_enriched_robertasentiment_ptbr2.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    agrupated_elements = {}
    all_matches = []
    for k, v in enumerate(data):
        if v["id"] in agrupated_elements:
            continue
        match_ele = []
        for k2, v2 in enumerate(data):
            if v2["id"] in agrupated_elements:
                continue
            if k != k2:
                score = fuzz.partial_ratio(v["nickname"] + v["name"], v2["nickname"] + v2["name"])
                if score > 80:
                    print(f"{v['nickname']} + {v['name']} - {v2['nickname']} + {v2['name']} - {score}")
                    agrupated_elements[v["id"]] = True
                    agrupated_elements[v2["id"]] = True
                    match_ele.append(v["id"])
                    match_ele.append(v2["id"])
        if len(match_ele) > 0:
            # print(match_ele)
            all_matches.append(match_ele)

    # save all matches to a file
    with open("data/influencers_matches.json", "w", encoding="utf-8") as file:
        json.dump(all_matches, file, ensure_ascii=False, indent=4)
