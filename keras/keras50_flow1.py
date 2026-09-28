# 48 카피
# 넘파이 형태일때 3차원데이터 4차원으로 변환하기
# 라벨링: 알파벳순

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

path = 'c:/Users/Admin/Desktop/Study_data/image/'

img = load_img(path+ 'face2.jpg', target_size = (200,200))  # load_img 이미지 한장 불러올때 편리

print(img)
# <PIL.Image.Image image mode=RGB size=200x200 at 0x1F24E159660>
print(type(img))  # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)  # (200, 200, 3) # 4차원 데이터로 reshape 필요
print(type(arr))  # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis = 0)  # 차원 증가
# print(arr)
print(arr.shape)   # (1, 200, 200, 3)

########## 요기부터 증폿이닷 ##########

datagen = ImageDataGenerator(
    rescale = 1./255,  # 전처리
    horizontal_flip = True, # 수평 뒤집기
    # vertical_flip = True,  # 수직 뒤집기 (상하반전)
    width_shift_range=0.1, # 평형이동
    # height_shift_range=0.1, 
    rotation_range=15, # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,
    # shear_range = 0.7, # 좌표하나를 규정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode = 'nearest'  # 이미지를 수평이동 했을때 빈공간을 가까운 값으로 채우겠다.

)

it = datagen.flow(arr, 
             batch_size= 1,
             )

print(it)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000002012E7A7F70>

# print(it.next())  # 파이썬 3.10까지,
print(next(it))     # 파이썬 3.11이후
print(next(it).shape)  # (1, 200, 200, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    # batch = it.next()
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(200,200,3)

    ax[i].imshow(batch)
    ax[i].axis('off')
plt.show()
