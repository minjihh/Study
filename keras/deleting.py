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
print(datasets)
print(datasets.DESCR)  # 판다스의 describe 있음
print(datasets.feature_names) # 판다스의 ".columns" 똑같은 기능

print(np.unique(y, return_counts=True))

from sklearn.preprocessing import OneHoeEncoder

ohe = OneHoeEncoder(sparse_output=False)

y = y.reshape(-1,1)