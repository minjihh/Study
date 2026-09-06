import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


## california

from sklearn.datasets import fetch_california

datasets = fetch_california()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)





## boston_housing

from tensorflow.keras.datasets import boston_housing

datasets = boston_housing.load_data()


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)


