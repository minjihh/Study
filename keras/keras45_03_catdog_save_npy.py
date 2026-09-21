# https://www.kaggle.com/datasets/tongpython/cat-and-dog/data    # 데이터셋
# 실습
# 맹그러봐!!
# 100x100 이하 사이즈로는 하지 말 것

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score


#1. 데이터

path_train = './_data/image/cat_dog/training_set'
path_test = './_data/image/cat_dog/test_set'

train_datagen = ImageDataGenerator(
    rescale = 1./255,  # 전처리
    # horizontal_flip = True, # 수평 뒤집기
    # vertical_flip = True,  # 수직 뒤집기 (상하반전)
    # width_shift_range=0.1, # 평형이동
    # height_shift_range=0.1, 
    # rotation_range=5, # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.2,
    # shear_range = 0.7, # 좌표하나를 규정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    # fill_mode = 'nearest'  # 이미지를 수평이동 했을때 빈공간을 가까운 값으로 채우겠다.
)

test_datagen = ImageDataGenerator(
    rescale = 1./255,  # test data는 scaling만 적용
)


xy_train = train_datagen.flow_from_directory(
    path_train, # 경로
    target_size=(200, 200),
    batch_size=4000,
    class_mode='binary', # 이진분류
    color_mode='rgb',  
    shuffle = True,
)
# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=1000,
    class_mode='binary', # 이진분류
    color_mode='rgb',  # 흑백
    shuffle = False,  # test에서는 shuffle 필요 없음 # shuffle default = True
)
# Found 2023 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)  # (5000, 200, 200, 3) (5000,)
print(x_test.shape, y_test.shape)  # (1000, 200, 200, 3) (1000,)


np_path = './_data/kaggle_cat_dog_npy/'

np.save(np_path + 'keras45_03_x_train.npy', arr=x_train)  # arr = xy_train[0][0]
np.save(np_path + 'keras45_03_y_train.npy', arr=y_train)  # arr = xy_train[0][1]
np.save(np_path + 'keras45_03_x_test.npy', arr=x_test)  # arr = xy_test[0][0]
np.save(np_path + 'keras45_03_y_test.npy', arr=y_test)  # arr = xy_test[0][1]

exit()

#2. 모델구성

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape= (200, 200, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Dense(40, activation = 'relu'))
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))


model.summary()


#3. 컴파일, 학습

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

path_pt = './_save/keras44/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'

file_path = "".join([path_pt, date,filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = file_path,
    save_best_only=True,
    verbose =1

)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 100,
    restore_best_weights= True,
    verbose =1
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs= 100000,
          batch_size = 16,
          callbacks = [es, mcp],
          verbose = 1,
          validation_split = 0.2)
end_time = time.time()

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score, 2))

print("걸린시간: ", round(end_time - start_time, 2) , '초')

# GAP
# acc_score:  0.83
# 걸린시간:  916.95 초

# Flatten
# acc_score:  0.76
# 걸린시간:  412.43 초

# GAP2
# acc_score:  0.85
# 걸린시간:  648.43 초

# Kaggle tensorflow
