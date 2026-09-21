from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
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


np_path = './_data/custom_image_npy/'

np.save(np_path + 'keras48_face2.npy', arr = arr)

print("numpy saved")


