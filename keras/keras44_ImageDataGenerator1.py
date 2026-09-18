import numpy as np
from keras.preprocessing.image import ImageDataGenerator

print(np.__version__)

train_datagen = ImageDataGenerator(
    rescale = 1./255,  # 전처리
    horizontal_flip = True, # 수평 뒤집기
    vertical_flip = True,  # 수직 뒤집기 (상하반전)
    width_shift_range=0.1, # 평형이동
    height_shift_range=0.1, 
    rotation_range=5, # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range = 1.2,
    shear_range = 0.7, # 좌표하나를 규정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode = 'nearest'  # 이미지를 수평이동 했을때 빈공간을 가까운 값으로 채우겠다.

)

test_datagen = ImageDataGenerator(
    rescale = 1./255,  # test data는 scaling만 적용
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, # 경로
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale',  # 흑백
    shuffle = True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale',  # 흑백
    shuffle = False,  # test에서는 shuffle 필요 없음 # shuffle default = True
)
# Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001DE7F187FA0>
# print(xy_train.next())   # 이터레이터의 첫번째를 보여줘!!
# print(xy_train.next())   # 이터레이터의 두번재를 보여줘!!

# print(xy_train[0])  # xy_train.next()의 첫번째와 동일
# print(xy_train[1])

# print(xy_train[0][0])  # 첫번째 배치의 x값만 출력
# print(xy_train[0][1])  # 첫번째 배치의 y값만 출력

# print(xy_train[0][0].shape)  # (10, 100, 100, 1)
# print(xy_train[0][1].shape)  # (10,)

# print(xy_train[16][0])  # 여기부터 에러. 이유는 160장이다!!! 배치는 10개니까

# print(type(xy_train))  # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # <class 'tuple'>
print(type(xy_train[0][0]))  # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))  # <class 'numpy.ndarray'>

