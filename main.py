import json
import os

import openai
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

_OPEN_AI_API_KEY = os.environ["OPEN_AI_API_KEY"]


class SentimentAnalysisResponse(BaseModel):
    """
    Represents a response from a sentiment analysis operation.

    Attributes:
        sentiment (str): The sentiment of the analyzed text.
        sentiment_score (float): The score indicating the sentiment intensity.
    """

    sentiment: str
    sentiment_score: float


class InfluerEnriched(BaseModel):
    id: str
    date: str
    name: str
    nickname: str
    rate: float
    when_worked: str
    work_description: str
    advice_description: str
    sentiment: str
    sentiment_score: float


def main():
    """main"""
    llm = OpenAI(model="gpt-4o-mini", api_key=_OPEN_AI_API_KEY)
    # I want to create a prompt that given a text will return a sentiment analysis

    # Prompt
    prompt = """
    Classifique o texto em very positive, positive, neutral, negative very negative. Também retorne um score associado a essa análise onde.
    0.5 a 1 very positive
    0.1 a 0.5 positivo
    -0.1 a 0.1 neutral
    -0.1 a -0.5 negativo
    -0.5 a -1 muito negativo

    Entrada:
    Text: Eu acho que a comida foi boa

    output:
    {
        "sentiment": "positive",
        "sentiment_score": 0.6
    }
    """

    with open("data/influencers_parsed_final.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    influencers_enriched = []
    for d in data:
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=d["work_description"] + " " + d["advice_description"]),
        ]
        print("processing")
        response = llm.as_structured_llm(SentimentAnalysisResponse).chat(messages)
        print("processed")
        resp_dict = response.dict()
        print(resp_dict)
        print(response)
        influencer = InfluerEnriched(
            id=d["id"],
            date=d["date"],
            name=d["name"],
            nickname=d["nickname"],
            rate=d["rate"],
            when_worked=d["when_worked"],
            work_description=d["work_description"],
            advice_description=d["advice_description"],
            sentiment=resp_dict["raw"]["sentiment"],
            sentiment_score=resp_dict["raw"]["sentiment_score"],
        )
        influencers_enriched.append(influencer.model_dump())

    # save the enriched influencers
    with open("data/influencers_enriched.json", "w", encoding="utf-8") as file:
        json.dump(influencers_enriched, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
