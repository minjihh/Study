import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd



path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path+ 'test.cvs', index_col=0)

print(train_csv.columns)
print(train_csv.info())

train_csv = train_csv.dropna()
x = train_csv.drop(['count'], axix=1)
y = train_csv(['count'])