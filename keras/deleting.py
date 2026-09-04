import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = './_data/kaggle_bike'


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)

submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'])
y = train_csv(['count'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

model = Sequential()
model.add(Dense(5, input_dim=11))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


model.compile()