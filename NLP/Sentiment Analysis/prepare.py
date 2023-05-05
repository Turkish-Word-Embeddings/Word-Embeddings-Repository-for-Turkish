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
            reviews.append((splitted[6], splitted[5])) # review, rating
    return reviews

negatives = read_txt('Turkish-Movie Dataset/movie_outerN.txt')
positives = read_txt('Turkish-Movie Dataset/movie_outerP.txt')
neutrals = read_txt('Turkish-Movie Dataset/movie_outerU.txt')

df1 = pd.DataFrame(
    {'review': [i[0] for i in negatives],
     "rating": [i[1] for i in negatives],
     "sentiment": 0
    })
df2 = pd.DataFrame(
    {'review': [i[0] for i in positives],
    "rating": [i[1] for i in positives],
    "sentiment": 1
    })
df3 = pd.DataFrame(
    {'review': [i[0] for i in neutrals],
    "rating": [i[1] for i in neutrals],
    "sentiment": 2
    })

df = pd.concat([df1, df2])
## Cross validation 
# create new column "kfold" and assign a random value
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

df3 = df3.sample(frac=1).reset_index(drop=True)
msk = np.random.rand(len(df3)) <= 0.8
train3 = df3[msk]
test3 = df3[~msk]
train3.to_csv("train_neutral.csv", index=False)
test3.to_csv("test_neutral.csv", index=False)