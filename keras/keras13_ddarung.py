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

train_csv = pd.read_csv(path + "train.csv", index_col=0)  # index_col 표기해서 데이터로 쓰지 않도록 함. // 모델 훈련전 인덱스 포함시키지 않도록 전처리
print(train_csv)  # [1459 rows x 11 columns]

# 출력시 제일 첫 컬럼은 실제 데이터에서는 존재하지 않는 컬럼이지만 판다스에서 보여줌
