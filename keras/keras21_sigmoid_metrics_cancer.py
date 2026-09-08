import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer  # 유방암관련 데이터셋 불러오기

#1. 데이터 

datasets = load_breast_cancer
print(datasets.DESCR)




x = datasets.data
y = datasets.target


