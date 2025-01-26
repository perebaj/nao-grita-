import json

import pandas as pd

with open("data/influencers_enriched3.json", "r", encoding="utf-8") as file:
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
print(df[["rank", "id", "nickname", "final_rating", "sentiment_score", "rate"]][:15])
