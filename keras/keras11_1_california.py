from sklearn.datasets import fetch_california_housing    # import 후에 ctrl + space 하면 어떤 리스트들이 있는지 확인 가능
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터

datasets = fetch_california_housing()  # 소문자로 시작하고 뒤에 () 붙으므로 함수임을 짐작할 수 있음
x = datasets.data
y = datasets.target
print(x.shape, y.shape)    # (20640, 8)  (20640,)

