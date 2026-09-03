# 데이터 -> 이상치, 결측치 처리후 사용
# 데이콘 설명 링크: https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


#1. 데이터

path = "./_data/ddarung/"

