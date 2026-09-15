import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


from sklearn.datasets import load_iris


#1. 데이터


datasets = load_breast_cancer()  # 이진분류
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

# scaling, one hot encoding

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)




# y = pd.get_dummies(y)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output= False)
y = y.reshape(-1,1)
ohe = ohe.fit(y)
y = ohe.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    random_state =1,
                                                    stratify=y,
                                                    )


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler

scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델 구성

model =Sequential()
model.add(Dense(4, input_dim = 30, activation = 'relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='sigmoid'))



input1 = Input(shape= (30,))
dense1 = Dense(4, activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(40, activatoin='relu')(drop1)
drop2 = Dropout(0.5)(dense2)
dense3 = Dense(40, activation = 'relu')(drop2)
drop3 = Dropout(0.5)(dense3)
output1 = Dense(40, activation='softmax')(drop3)

model = Model(inputs = input1, outputs = output1)



model.summary()

#3. 컴파일 훈련

model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')

pt_path = './_save/keras34/'
filename = '{epoch:4d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)






start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=32,
                 verbose=1,
                 callbacks=[es, mcp])
end_time = time.time()

model.save(path + 'keras29_m.keras')
model.save_weights(path + )

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print("mse: ", loss)
r2 = r2_score(y_test, y_pred)
print("r2_score: ", r2)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse = RMSE(y_test, y_pred)
print("rmse: ", rmse)




