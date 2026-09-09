from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd


datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=1, stratify=y)

model = Sequential()
model.add(Dense(5, input_dim = 64, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=10,
    restore_best_weights=True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100000, batch_size=16, validation_split=0.2, callbacks=[es])








# acc: 1.0