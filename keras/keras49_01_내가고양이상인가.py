"""
개고양이 가중치를 가져와서 모델 완성
데이터는 개 고양이 npy데이터로 사용
내사진도 npy불러와서 predict만 하면 되것지요.
끗!!!
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score

# ## 이미지 직접 불러오기 ###
# img_path = 'c:/Users/Admin/Desktop/Study_data/image/'
# img = load_img(img_path+ 'images.jpg', target_size = (200,200))




### 이미지 넘파이로 불러오기 ###   아래 넘파이파일은 rescaling 안된 데이터

np_path = './_data/custom_image_npy/'
img = np.load(np_path + 'keras48_me5.npy')

img = img/255.


#### model ####

model_path = './_save/keras45/'
model = load_model(model_path + '0921_08250052-0.4826-catdog.keras')


result = model.predict(img)
print("result: ", result)
result = np.round(result)

print("결과: ", result)


# image1  강아지
# result:  [[0.7802793]]
# 결과:  [[1.]]  개

# image 2 강아지
# result:  [[0.9634653]]
# 결과:  [[1.]] 개

# image 3 고양이
# result:  [[0.17383644]]
# 결과:  [[0.]] 고양이

# image 4 호랑이
# result:  [[0.11554857]]
# 결과:  [[0.]] 고양이

# image 5 곰
# result:  [[0.913433]]
# 결과:  [[1.]]  개