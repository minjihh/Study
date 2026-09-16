from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

# 실습 맹그러봐요

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(np.unique(y_train, return_counts=True))



# one hot encoding
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


#2. 모델구성

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(10, (2,2), activation ='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(12, (3,3), activation ='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation ='relu'))
model.add(Conv2D(16, (2, 2), activation = 'relu'))  
model.add(Conv2D(16, (2, 2), activation = 'relu'))  
model.add(Conv2D(16, (2, 2), activation = 'relu')) 


model.add(Flatten())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(10))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 250,
    restore_best_weights= True,
    verbose = 1
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 100000,
          batch_size = 128,
          verbose = 1,
          validation_split=0.2,
        #   metrics = ['acc'],
          callbacks = [es]
          )
end_time = time.time()




loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis =1)

acc_score = accuracy_score(y_pred, y_test) 

print("acc_score: ", acc_score)

print("걸린시간: ", round(end_time - start_time, 2), '초')


import matplotlib.pyplot as plt

# plt.imshow(x_train[50],) # 'gray')  # 원래 grayscale 이미지에서 'gray' 설정 안했다고해서 color 이미지라고 할 순 없음
# plt.show()