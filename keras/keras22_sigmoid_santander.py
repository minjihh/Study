# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# path = "./_data/kaggle_santander/"
path = "c:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')


print(train_csv.shape)  # (200000, 202)


