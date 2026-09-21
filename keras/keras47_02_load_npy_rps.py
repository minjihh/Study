import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # 이진분류를 softmax로, 목표 acc 1.0





#1. 데이터


data_path = 'c:/Users/Admin/Desktop/Study_data/image/rps/'

datagen = ImageDataGenerator(
    rescale = 1/255.
)

xy = datagen.flow_from_directory(data_path,
                                 target_size = (200, 200),
                                 batch_size = 2000,
                                 class_mode = 'categorical',
                                 color_mode = 'rgb',
                                 shuffle = True,
)

# x = xy[0][0]
# y = xy[0][1]

# print(x.shape, y.shape)  # (100, 200, 200, 3) (100, 3)

# print(np.unique(y, return_counts = True))

np_path = './_data/rps_npy/'
# np.save(np_path + 'keras46_rps_x.npy', arr=x)
# np.save(np_path + 'keras46_rps_y.npy', arr=y)

x = np.load(np_path + 'keras46_rps_x.npy')
y = np.load(np_path + 'keras46_rps_y.npy')



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    random_state = 1,
                                                    )


print(x_train.shape, y_train.shape)  #  (70, 200, 200, 3) (70, 3)


#2. 모델구성

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape = (200,200,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))


model.add(Dense(64, activation = 'relu'))
model.add(Flatten())
# model.add(MaxPooling2D())   #     ValueError: Shapes (None, 3) and (None, 100, 100, 3) are incompatible
model.add(Dense(3, activation = 'softmax'))

model.summary()



#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor ='val_loss',
    mode = 'auto',
    verbose = 1,
    patience=200,
    restore_best_weights = True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 10000,
          batch_size = 32,
          validation_split=0.2,
          verbose =1,
          callbacks = [es]
          )
end_time = time.time()


#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)

acc_score =accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score, 2))
print("걸린시간: ", round(end_time - start_time, 2))


