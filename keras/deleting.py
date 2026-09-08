from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import time

datasets = fetch_california_housing

x = datasets.data
y = datasets.target

x_train, y_train, x_test, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
x_val, y_val, x_test, y_test = train_test_split(x_test, y_test, test_size =0.5, random_state=1)


model = Sequential()


model.compile(loss)