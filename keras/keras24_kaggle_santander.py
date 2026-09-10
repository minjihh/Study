# 이진분류 데이터 다중분류 방식으로 풀어보기
# 이진분류로 했을때랑 비교해보기

from tensorflow.keras.models import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time


