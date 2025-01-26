import os
import openai
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel
from llama_index.core.llms import ChatMessage

_OPEN_AI_API_KEY = os.environ["OPEN_AI_API_KEY"]


class SentimentAnalysisResponse(BaseModel):
    """
    Represents a response from a sentiment analysis operation.

    Attributes:
        sentiment (str): The sentiment of the analyzed text.
        sentiment_score (float): The score indicating the sentiment intensity.
        text (str): The original text that was analyzed.
    """

    sentiment: str
    sentiment_score: float
    text: str


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
        "text": "I think the food was okay",
        "sentiment": "positive",
        "sentiment_score": 0.6
    }
    """

    # Text

    text = "um querido e a equipe super pronta pra flexibilizações. Cumpre agenda, mesmo que sejam ajustes finíssimos, são compreensivos."

    messages = [ChatMessage(role="system", content=prompt), ChatMessage(role="user", content=text)]

    # use the llama index to call the sentiment analysis function
    # response = llm.chat(messages)
    response = llm.as_structured_llm(SentimentAnalysisResponse).chat(messages)
    print(response)


if __name__ == "__main__":
    main()
