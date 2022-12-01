import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/creditcard.csv")
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

np.random.seed(36)

train, test = train_test_split(df, test_size=0.25)
train.to_csv("./data/creditcard_train.csv", index=False)
test.to_csv("./data/creditcard_test.csv", index=False)

