# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # 이진분류를 softmax로, 목표 acc 1.0

#1. 데이터


data_path = 'c:/Users/Admin/Desktop/Study_data/image/faces/'

datagen = ImageDataGenerator(
    rescale = 1/255.
)

xy = datagen.flow_from_directory(data_path,
                                 target_size = (200, 200),
                                 batch_size = 10000,
                                 class_mode = 'binary',
                                 color_mode = 'rgb',
                                 shuffle = True,
)

x = xy[0][0]
y = xy[0][1]

print(x.shape, y.shape)  # (100, 200, 200, 3) (100, 3)

print(np.unique(y, return_counts = True))

np_path = './_data/gender_npy/'
np.save(np_path + 'keras46_gender_x.npy', arr=x)
np.save(np_path + 'keras46_gender_y.npy', arr=y)

# x = np.load(np_path + 'keras46_rps_x.npy')
# y = np.load(np_path + 'keras46_rps_y.npy')



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    random_state = 1,
                                                    stratify = y
                                                    )


print(x_train.shape, y_train.shape)  #  (70, 200, 200, 3) (70, 3)


#2. 모델구성

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape = (200,200,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
 

model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

# model.add(Dropout(0.2))

# model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
# model.add(Dropout(0.2))



# model.add(Flatten())
model.add(GlobalAveragePooling2D())   
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()



#3. 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

path_pt = './_save/keras46/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}-gender.keras'

file_path = "".join([path_pt, date,filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = file_path,
    save_best_only=True,
    verbose =1

)
es = EarlyStopping(
    monitor ='val_loss',
    mode = 'auto',
    verbose = 1,
    patience=100,
    restore_best_weights = True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 10000,
          batch_size = 16,
          validation_split=0.2,
          verbose =1,
          callbacks = [es, mcp]
          )
end_time = time.time()




#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)


acc_score =accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score, 2))
print("걸린시간: ", round(end_time - start_time, 2))


# acc_score:  0.7
# 걸린시간:  46.57

# acc_score:  0.73
# 걸린시간:  108.42