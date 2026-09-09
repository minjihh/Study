from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

datasets = fetch_covtype()
print(fetch_covtype.info())

x = datasets.data
y = datasets.target
print(x.shape, y.shape)





# 시간재기
# 배치 크게
# acc 0.93