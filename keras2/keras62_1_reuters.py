from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, SpatialDropout1D
import time
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,  # 단어사전의 갯수, 빈도수의 갯수가 높은 단어 순으로 1000개 뽑겠다.
    # maxlen = 1000,    # default : 전체데이터 
    test_split = 0.2,
)


print(x_train)
print(x_train.shape, y_train.shape)  # (8982,) (8982,)
print(x_test.shape, y_test.shape)    # (2246,) (2246,)
print(y_train)
# y데이터의 클래스 확인하기 (np.unique, pandas의 value counts)
#1. numpy의 np.unique
print(np.unique(y_train))

#2. pandas의 value_counts
# df = pd.DataFrame(y_train)
# print(df.value_counts)

print(type(x_train)) # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>   (numpy안에 list가 들어가있는 상태)
print(len(x_train[0]), len(x_train[1]))  # 87 56

print("뉴스기사의 최대길이 : ", max( len(i) for i in x_train))  # 2376
print("뉴스기사의 최소길이 : ", min( len(i) for i in x_train))  # 13
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/len(x_train))  # 145.53

# 전처리 (패드 시퀀스)
x_train_padded = pad_sequences(x_train,
                        padding = 'pre',  # pre: 앞쪽을 0으로 채움, post: 뒤쪽을 0으로 채움
                        maxlen = 50,  # maxlen보다 길경우: default는 앞쪽부터 잘려나감
                        truncating = 'post', 
                        )

print(x_train_padded.shape)  # (8982, 5)

x_test_padded = pad_sequences(x_test,
                        padding = 'pre',  # pre: 앞쪽을 0으로 채움, post: 뒤쪽을 0으로 채움
                        maxlen = 50,  # maxlen보다 길경우: default는 앞쪽부터 잘려나감
                        truncating = 'post', 
                        )

print(x_test_padded.shape)  # (2246, 5)

# y 원핫
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohe = ohe.fit(y_train)
ohe_y_train = ohe.transform(y_train)
ohe_y_test = ohe.transform(y_test)

print(ohe_y_train.shape, ohe_y_test.shape)  # (8982, 46) (2246, 46)


x_train_padded = x_train_padded.reshape(x_train_padded.shape[0], x_train_padded.shape[1],1)
x_test_padded = x_test_padded.reshape(x_test_padded.shape[0], x_test_padded.shape[1],1)

print(x_train_padded.shape, x_test_padded.shape)   # (8982, 5, 1) (2246, 5, 1)



#2. 모델 구성
model = Sequential()
model.add(Embedding(1000, 500, mask_zero = True)) 
# mask_zero=True -> pad_seuence에서 추가시킨 0을 마스킹처리, index 0이 vocabulary에서 사용되지 않음
# model.add(SpatialDropout1D(0.2))
model.add(LSTM(10, input_shape= (50,1)))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(46, activation = 'softmax'))


#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

learning_rate = 0.001
model.compile(loss= 'categorical_crossentropy', optimizer = Adam(learning_rate = learning_rate))

### mcp 파일명 ###
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras62/'
filename = '{epoch:04d}-{val_loss:.4f}_reuters.keras'
filepath = "".join([pt_path, "k62_", date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = filepath,
    save_best_only=True,
    verbose=1
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 50,
    restore_best_weights= True,
    verbose = 1
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 5,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)

start_time = time.time()
model.fit(x_train_padded, ohe_y_train,
          epochs= 500,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es, rlr],
          )
end_time = time.time()

#4. 평가 예측

loss = model.evaluate(x_test_padded, ohe_y_test)
print("loss: ", loss)

y_pred = model.predict(x_test_padded)
y_pred = np.argmax(y_pred, axis =1)
y_test = np.argmax(ohe_y_test, axis = 1)


acc_score = accuracy_score(y_test, y_pred)


print("걸린시간: ", round(end_time - start_time, 2))
print("loss: ", round(loss, 2))
print("acc: ", round(acc_score,2))

### acc 0.67 이상!
# 임베딩차원과 maxlen을 조절해서 성능을 높여볼 수 있다.


# dropout 추가, 100 epochs, embedding(1000,500), rlr 추가, maxlen 50
# 걸린시간:  70.12
# loss:  1.38
# acc:  0.69

# dropout 추가, 100 epochs, embedding(1000,500), rlr 추가, maxlen 50, mask_zero = True
# 걸린시간:  990.9
# loss:  1.42
# acc:  0.67