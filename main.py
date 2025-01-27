import json
import os

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
    llm = OpenAI(model="gpt-3.5-turbo", api_key=_OPEN_AI_API_KEY)
    # I want to create a prompt that given a text will return a sentiment analysis

    # Prompt
    prompt = """
    Classifique o texto em uma das seguintes categorias de sentimento: very positive, positive, neutral, negative, ou very negative. Além disso, forneça um score de sentimento associado, com precisão de até 5 casas decimais, seguindo as regras abaixo:

    - 0.50000 a 1: very positive
    - 0.10000 a 0.50000: positive
    - -0.10000 a 0.10000: neutral
    - -0.10000 a -0.50000: negative
    - -0.50000 a -1: very negative

    Garanta que o score fornecido tenha exatamente 5 casas decimais para aumentar a precisão da análise.

    Entrada:
    Texto: Eu acho que a comida foi boa

    Saída esperada:
    {
        "sentiment": "positive",
        "sentiment_score": 0.0000
    }

    """

    with open("data/influencers_parsed_final.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    influencers_enriched = []
    errors_id = []
    for d in data:
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=d["work_description"] + " " + d["advice_description"]),
        ]
        print("processing")
        try:
            response = llm.as_structured_llm(SentimentAnalysisResponse).chat(messages)
            resp_dict = response.dict()
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
            print(resp_dict["raw"]["sentiment"], resp_dict["raw"]["sentiment_score"])
            influencers_enriched.append(influencer.model_dump())
        except Exception as e:
            print("Error processing influencer {}. Error. {}".format(d["id"], e))
            errors_id.append(d["id"])
            continue
        print("processed")

    # save the enriched influencers
    with open("data/influencers_enriched_gpt3.5.json", "w", encoding="utf-8") as file:
        json.dump(influencers_enriched, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
