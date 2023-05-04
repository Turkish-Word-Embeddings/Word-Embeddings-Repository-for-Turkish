import numpy as np
import pandas as pd

# read txt dataset
def read_txt(path):
    # create a dataframe consisting of review and sentiment
    reviews = []
    with open(path, 'r', encoding="utf-8") as f:
        data = f.readlines()
        for line in data[1:]:
            splitted = line.split("\t")
            reviews.append(splitted[6])
    return reviews

negatives = read_txt('Turkish-Movie Dataset/movie_outerN.txt')
positives = read_txt('Turkish-Movie Dataset/movie_outerP.txt')

df1 = pd.DataFrame(
    {'review': negatives,
     "sentiment": 0
    })
df2 = pd.DataFrame(
    {'review': positives,
    "sentiment": 1
    })
df = pd.concat([df1, df2])
## Cross validation 
# create new column "kfold" and assign a random value
df['kfold'] = -1  # all rows have df['kfold'] = -1

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
df.head(5)
# split the df to train and test by 0.8 ratio
msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
test = df[~msk]
# save train and test to csv
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
