import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_iris()
print(datasets.info())
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y = y.reshape(-1,1)
# y = y.reshape(len(y),1)
ohe_y = ohe.fit_transform(y)



y = pd.get_dummis(y)

from sklearn.preprocessing import OneHotEncoder
ohd = OneHotEncoder(sparse_output=False)
y = y.reshape(-1,1)
y = y.reshape(len(y),1)
y = ohd.fit_transform(y)

exit()



y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis = 1)
y_test = np.argmax(y_test, axis = 1)






