import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer  # 유방암관련 데이터셋 불러오기

#1. 데이터 

datasets = load_breast_cancer()

### 보통은 데이터 판다스로 많이 불러오기 때문에 아래 2개랑 비슷한 역할하는 pandas 문법을 주로 사용 ###
print(datasets.DESCR)  # Number of Attributes: 30 numeric 
print(datasets.feature_names)
######################

# x = datasets.data   # datasets.data == datasets['data']
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)



