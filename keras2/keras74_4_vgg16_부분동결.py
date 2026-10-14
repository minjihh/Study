# 74 - 3 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(
    weights = 'imagenet',
    include_top=False,
    input_shape = (32, 32, 3),
)

# model.trainable = False



model.layers[11].trainable = False
model.summary()

# exit()

import pandas as pd  # pandas에어는 DataFrame(행렬)과 Series(벡터) 두가지 데이터 타입밖에 없음, tensor는 없다.
pd.set_option('max_colwidth', None)  # 문자열을 제한없이 보여주겠다.

layers = [
    (type(layer).__name__, layer.name, layer.trainable) 
    for layer in model.layers
    
]

results = pd.DataFrame(
    layers, columns= ['Layer Type', 'Layer Name', 'Layer Trainable']
    )

print(results)

#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True


