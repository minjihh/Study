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
# custom

np_path = './_data/custom_image_npy/'
img = np.load(np_path + 'keras48_face2.npy')

img = img/255.


# img = img/255.


#### model ####

model_path = './_save/keras46/'
model = load_model(model_path + '0921_15440017-0.4623-gender.keras')


result = model.predict(img)
print("result: ", result)
result = np.round(result)

print("결과: ", result)

# 여자이미지
# result:  [[0.14189598]]
# 결과:  [[0.]]

# 여자이미지
# result:  [[0.04939599]]
# 결과:  [[0.]]

