import numpy as np
import pandas as pd
import os

##
# This script reads the Turkish Twitter dataset (its original form, which is not available here) and converts it to a CSV file.
##

# Define the directories
base_dir = "Turkish-Twitter Dataset"
subfolders = ["test", "training"]
sentiments = ["negative", "neutral", "positive"]

# Initialize two empty lists
data_train = []
data_test = []

label_encoder = {"positive": 1, "negative": 0, "neutral": 2}

# Loop through the directories and read the files
for subfolder in subfolders:
    for sentiment in sentiments:
        path = os.path.join(base_dir, subfolder, sentiment)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                tweet = f.read()
                if label_encoder[sentiment] != 2:
                    if subfolder == "training":
                        data_train.append(
                            {"review": tweet, "sentiment": label_encoder[sentiment]}
                        )
                    else:
                        data_test.append(
                            {"review": tweet, "sentiment": label_encoder[sentiment]}
                        )

# Convert the lists to DataFrames and save them to CSV files
df_train = pd.DataFrame(data_train)
df_train.to_csv("twitter_train_data.csv", index=False)

df_test = pd.DataFrame(data_test)
df_test.to_csv("twitter_test_data.csv", index=False)
