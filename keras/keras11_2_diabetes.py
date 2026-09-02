from sklearn.datasets import fetch_california_housing, load_diabetes    # import 후에 ctrl + space 하면 어떤 리스트들이 있는지 확인 가능
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (442,10) (442,)

# [실습] 맹그러봐요!!
