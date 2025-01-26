import json
import uuid

from pydantic import BaseModel


class Influencer(BaseModel):
    """
    Represents an influencer.

    Attributes:
        id (str): The ID of the comment
        date (str): When the comment was made.
        name (str): The name of the influencer.
        nickname (str): The nickname of the influencer.
        rate (float): The rate of the influencer.
        when_worked (str): The period when marketing agency worked with the influencer.
        work_description (str): The description of the influencer's work.
        advice_description (str): An advice to work with the influencer.
    """

    id: str
    date: str
    name: str
    nickname: str
    rate: float
    when_worked: str
    work_description: str
    advice_description: str


# Specify the file path
FILE_PATH = "data/influencers-parsed.json"


def parse_influencers():
    # Read the JSON file
    with open(FILE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Access the data
    # Example: Print the contents of the JSON file

    # Create a list to store the influencers
    influencers = []
    exceptions = []
    for page in data["pages"]:
        for item in page["items"]:
            for row in item["rows"]:
                try:
                    influencer = Influencer(
                        id=row[0],
                        date=row[1],
                        name=row[2],
                        nickname=row[3],
                        rate=row[4],
                        when_worked=row[5],
                        work_description=row[6],
                        advice_description=row[7],
                    )
                    influencers.append(influencer)
                except Exception as e:
                    print(e)
                    print(row)
                    exceptions.append(row)

    # save in json file
    with open("data/influencers.json", "w", encoding="utf-8") as file:
        json.dump([influencer.dict() for influencer in influencers], file, indent=4, ensure_ascii=False)

    with open("data/exceptions.json", "w", encoding="utf-8") as file:
        json.dump(exceptions, file, indent=4, ensure_ascii=False)


def parse_exception():
    """It only change the order of the fields that will be encoded in the json file."""
    # Read the JSON file
    with open("data/exceptions.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    exceptions_influencers = []
    for row_num, row in enumerate(data):
        try:
            influencer = Influencer(
                date=row[0],
                name=row[1],
                nickname=row[2],
                rate=row[3],
                when_worked=row[4],
                work_description=row[5],
                advice_description=row[6],
                id=uuid.uuid4().int,
            )
            exceptions_influencers.append(influencer)
        except Exception as e:
            print(e)
            print(f"Row number: {row_num}. Data: {row}")

    # save in json file
    with open("data/exceptions-influencers.json", "w", encoding="utf-8") as file:
        json.dump([influencer.dict() for influencer in exceptions_influencers], file, indent=4, ensure_ascii=False)


def replace_idint_byuuid():
    """It only change the order of the fields that will be encoded in the json file."""
    # Read the JSON file
    with open("data/influencers.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    influencers = []
    for influencer in data:
        # save a copy of the influencer
        i = influencer.copy()
        uuidd = uuid.uuid4()
        i["id"] = str(uuidd)
        i["rate"] = influencer["rate"]
        i["name"] = influencer["name"]
        i["nickname"] = influencer["nickname"]
        i["when_worked"] = influencer["when_worked"]
        i["work_description"] = influencer["work_description"]
        i["advice_description"] = influencer["advice_description"]
        i["date"] = influencer["date"]

        influencers.append(i)

    # save in json file
    with open("data/influencers-finisheds.json", "w", encoding="utf-8") as file:
        json.dump(influencers, file, indent=4, ensure_ascii=False)


replace_idint_byuuid()
