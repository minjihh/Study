import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer  # 유방암관련 데이터셋 불러오기


datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
x = datasets['data']
y = datasets.target

print(x.shape, y.shape)
print(type(x))

print(np.unique(y))
print(np.unique(y, return_counts=True))

print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())

print(np.unique(y, return_counts=True))

print(datasets.DESCR)
print(datasets.featre_names)

print(np.unique(y, return_counts = True))

print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())
