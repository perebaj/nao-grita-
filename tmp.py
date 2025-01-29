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


# Exemplo de uso
example_reviews = [
    # {"rate": 2.0, "sentiment": "negative", "sentiment_score": 0.97438},
    # {"rate": 10.0, "sentiment": "positive", "sentiment_score": 0.98364},
    {"rate": 9.0, "sentiment": "negative", "sentiment_score": 0.34793},
]

rating = calculate_influencer_rating(example_reviews)
print(rating)
