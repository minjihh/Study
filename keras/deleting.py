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

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

y = pd.get_dummis(y)

from sklearn.preprocessing import OneHotEncoder
ohd = OneHotEncoder(sparse_output=False)




