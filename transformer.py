import datetime
import json
import random
import uuid

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


def calculate_influencer_rating(reviews):
    """
    Calcula o rating final do influenciador baseado em avaliações, sentimento e scores.

    Args:
        reviews: Lista de dicionários contendo as avaliações

    Returns:
        float: Rating final na escala 0-5
    """
    if not reviews:
        return 0.0

    def normalize_sentiment_score(sentiment, score):
        # Inverte o score para sentimentos negativos
        if sentiment == "negative":
            return 1 - score
        elif sentiment == "neutral":
            return 0.5
        else:  # positive
            return score

    total_weight = 0
    weighted_sum = 0

    for review in reviews:
        # Normaliza o rate original para escala 0-1
        normalized_rate = review["rate"] / 10.0

        # Calcula o score de sentimento normalizado
        sentiment_normalized = normalize_sentiment_score(review["sentiment"], review["sentiment_score"])

        # Peso baseado na confiança do modelo de sentimento
        weight = review["sentiment_score"]

        # Combina rate e sentimento com pesos iguais
        review_score = (normalized_rate + sentiment_normalized) / 2

        weighted_sum += review_score * weight
        total_weight += weight

    # Calcula média ponderada final
    final_score = weighted_sum / total_weight if total_weight > 0 else 0

    # Converte para escala 0-5
    return round(final_score * 5, 2)


def call_calculate_final_rating():
    """
    Calls the calculate_influencer_rating function for each influencer in the data.
    Saves the results to a file.

    Parameters:
    None

    Returns:
    None
    """
    with open("data/influencers_enriched_robertasentiment_ptbr.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    calculate_influencer_rating(data)

    # save the results to a file

    with open("data/influencers_enriched_robertasentiment_ptbr2.json", "w", encoding="utf-8") as file:
        current_datetime = datetime.datetime.now()
        random_number = random.randint(1000, 9999)
        file_name = f"data/influencers_enriched_robertasentiment_ptbr2_{current_datetime.strftime('%Y%m%d%H%M%S')}_{random_number}.json"

    # Save the results to the generated file name
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


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


def group_reviews():
    with open("data/influencers_matches.json", "r", encoding="utf-8") as file:
        matches = json.load(file)

    with open("data/influencers_enriched_robertasentiment_ptbr.json", "r", encoding="utf-8") as file:
        influencers = json.load(file)

    all_reviews = []
    for match in matches:
        reviews = []
        for id in match:
            for influencer in influencers:
                if influencer["id"] == id:
                    reviews.append(influencer)
                    break
        # create a new dictionary with the grouped reviews and assign a uuid to it
        uid = str(uuid.uuid4())
        reviews = {"uuid": uid, "reviews": reviews}
        final_rating = calculate_influencer_rating(reviews["reviews"])
        reviews["final_rating"] = final_rating
        print(reviews)
        all_reviews.append(reviews)

    # find the reviews that were not grouped
    for influencer in influencers:
        if influencer["id"] not in [review for match in matches for review in match]:
            uid = str(uuid.uuid4())
            reviews = {"uuid": uid, "reviews": [influencer]}
            final_rating = calculate_influencer_rating(reviews["reviews"])
            reviews["final_rating"] = final_rating
            all_reviews.append(reviews)

    with open("data/influencers_grouped_reviews3.json", "w", encoding="utf-8") as file:
        json.dump(all_reviews, file, ensure_ascii=False, indent=4)


def rank_influencer():
    with open("data/influencers_grouped_reviews3.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    influencers_analysis = []
    for influencer_reviews in data:
        uid = influencer_reviews["uuid"]
        final_rating = influencer_reviews["final_rating"]

        print(f"UUID: {uid} - Final Rating: {final_rating}")

        influencer_reviews_df = pd.DataFrame(influencer_reviews["reviews"])
        #arbitrarily select the last review to get the nickname
        last_review = influencer_reviews_df.iloc[-1]

        influencers_analysis.append(
            {
                "uuid": uid,
                "final_rating": final_rating,
                "reviews": influencer_reviews["reviews"],
                "nickname": last_review["nickname"],
                "name": last_review["name"],
                "num_reviews": len(influencer_reviews_df),
                "num_negative": len(influencer_reviews_df[influencer_reviews_df["sentiment"] == "negative"]),
                "num_neutral": len(influencer_reviews_df[influencer_reviews_df["sentiment"] == "neutral"]),
                "num_positive": len(influencer_reviews_df[influencer_reviews_df["sentiment"] == "positive"]),
                "avg_rate": influencer_reviews_df["rate"].mean(),
            }
        )

    # save the results to a file
    with open("data/influencers_ranked3.json", "w", encoding="utf-8") as file:
        json.dump(influencers_analysis, file, ensure_ascii=False, indent=4)


rank_influencer()
