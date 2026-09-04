import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

path = "./_data/ddarung/"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path +"test.csv", index_col=0)

submission = pd.read_csv(path + "submission.csv", index_col=0)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())


train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis = 1)

y = train_csv['count']