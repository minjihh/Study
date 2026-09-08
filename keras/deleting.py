import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size =0.7, random_state=1)

model = Sequential()
model.add(Dense(5), input_dim=200, activation ='relu')
model.add(Dense(7), activation ='relu')
model.add(Dense(1), activation ='sigmoid')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True,
)

model.compile(loss='mse', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs = 10, batch_size=16, validation_data=(x_val, y_val), callbacks=[es])

loss = model.evaluate(x_val, y_val)
print("loss: ", loss)
y_pred = model.predict(test_csv)

submission['target'] = y_pred
submission.to_csv(path + 'submission_0909_3.csv')



print(train_csv.info())
print(np.unique(y, return_counts=True))
print(pd.DataFrame(y_pred).value_counts())
print(pd.Series(y).value_counts())