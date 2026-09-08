# load_breast_cancer 데이터 불러오기
# type()를 통해 데이터 형태 확인

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
x = datasets['data']  # datasets은 딕셔너리형태의 데이터였다.
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))   # <class 'numpy.ndarray'>  type: data 형태 보여줌
# -> pandas 데이터도 numpy로 만들어져 있다.


# 분류형 모델 꼭 y를 확인해야함 -> label 종류 개수 확인필요
print(y)
print(np.unique(y))  # [0 1]
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357]))
# 분류/범주형 데이터 받으면 꼭 확인
# 범주형 데이터에서는 데이터 개수보다는 데이터 balance 좋은것이 좋다. 



