import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import time



random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


from keras.applications import VGG16
from tensorflow.keras.datasets import cifar100


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(np.unique(y_train, return_counts=True)) 

#2. 모델

vgg16 = VGG16(
    weights = 'imagenet',  
    include_top = False,   # False로 하면 input_shape변경 가능, Flatten과 하단의 FC layer 사라짐
    input_shape = (32, 32, 3),
)

vgg16.trainable = False
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

model.summary()

#3. 컴파일 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['acc'])


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=100,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 100,
          batch_size = 128,
          validation_split=0.2,
          verbose = 1,
          callbacks = [es]
          )
end_time = time.time()


#4. 평가 예측
print("=============== model.evaluate ===============")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_pred = model.predict(x_test)

# y_test = np.argmax(y_test, axis = 1)
y_pred = np.argmax(y_pred, axis = 1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score,2))

print("걸린시간: ", round(end_time - start_time, 2), '초')



# 실습 #
# 비교할거
#1. 이전에 본인이 한 최상의 결과.


#2. 가중치를 동결하고 훈련시켰을때, trainable=False
# loss:  2.9260871410369873
# acc:  0.3181000053882599
# acc_score:  0.32
# 걸린시간:  41.29 초

# loss:  2.9664182662963867
# acc:  0.3255999982357025
# acc_score:  0.33
# 걸린시간:  442.99 초

# dense layer 변경, relu 추가


#3. 가중치를 동결하지 않고 훈련시켰을때, trainable=True
# loss:  4.605472087860107
# acc:  0.009999999776482582
# acc_score:  0.01
# 걸린시간:  106.39 초

#4. 시간까지 비교

