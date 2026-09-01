# 데이터를 받으면 shape부터 확인해야한다. // 모델 구성단계에서 input_dim을 설정해줄때 shape 확인후에 설정

import numpy as np

x1 = np.array([1,2,3])  # shape: (3,) 벡터 3개
print("x1 = ", x1.shape) 

x2 = np.array([[1,2,3]])  # shape: (1,3) 행렬
print("x2 =", x2.shape)

x3 = np.array([[1,2], [3,4]]) # shape: (2,2) 행렬
print("x3 =", x3.shape)

x4 = np.array([[1,2], [3,4], [5,6]]) # shape: (3,2) 행렬
print("x4 =", x4.shape)

# x5 = np.array([[1,2], [3,4], [5,6,7]])  # 이런 데이터는 없음. 에러 (결측치가 들어갔을 가능성)

x6 = np.array([[1,2], [3,4], [5,6,]])  # shape: (3,2) 행렬 # ,로 뒤에 데이터가 더 올수 있도록 여지를 남겨둠. 데이터 이상 없음

x7 = np.array([[[1,2], [3,4], [5,6,]]])  # shape: (1,3,2) 텐서 
print("x7 =", x7.shape)

x8 = np.array([[[1,2,], [3,4]], [[5,6,],[7,8]]]) # shape: (2,2,2) 텐서
print("x8 =", x8.shape)

x9 = np.array([[[[[1,2,3,4,5], [6,7,8,9,10]]]]]) # shape: (1,1,1,2,5) 5차원 텐서
print("x9 =", x9.shape)

x10 = np.array([[[1,2,3]],[[4,5,6]]])  # shape: (2,1,3) 텐서, 실무에서는 잘 없는 데이터
print("x10 =", x10.shape)

x11 = np.array([[[[1]]], [[[2]]]]) # shape: (2,1,1,1) 4차원 텐서
print("x11 =", x11.shape)

