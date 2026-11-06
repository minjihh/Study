# m04 -1 카피
# m02 00 iris 카피

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings('ignore')


#1. 데이터
iris = load_iris(return_X_y = True)
# print(iris)
cancer = load_breast_cancer(return_X_y = True)
wine = load_wine(return_X_y = True)
digits = load_digits(return_X_y = True)

datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']



#2. 모델
all = all_estimators(type_filter='classifier')

n_splits = 5
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state = 333)


start = time.time()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        shuffle = True,
        random_state = 333,
        stratify = y,
        train_size = 0.8,
        )
        
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("=============", data_name[index], "=================")
    model_name=[]
    model_acc=[]


    # all에 2개 들어가 있으므로 앞에꺼는 name, 2번째꺼는 model로 빼옴
    for name, model in all:
        # print(name)
        try:
            model = model()   # 인스턴스 생성도 try 안에서 (estimator 필수 인자 모델은 여기서 오류)

            #3. 훈련
            # model.fit(x_train, y_train)
            scores = cross_val_score(model, x_train, y_train, cv = kfold)

            #4. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv = kfold)

            acc = np.round(accuracy_score(y_test, y_pred), 4)
            model_name.append(name)
            model_acc.append(acc)
        
            # print(acc)
            # exit()

        except:
            # print(name, "모델 오류")
            continue

    model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.argmax(model_acc)
    print('최고 모델: ', model_name[max_index], "ACC: ", model_acc[max_index])

end = time.time()

print("걸린시간 :", round(end-start, 2), '초')




exit()


#3,4 훈련(컴파일 없음), 평가
scores= cross_val_score(model, x_train, y_train, cv = kfold)  # cross_val_score가 model.fit까지 시킴

print('acc : ', scores, '\n평균 acc :', round(np.mean(scores),2))
# 분류: acc, 회귀: R2
# acc :  [0.93333333 0.9        1.         0.96666667 0.93333333] 
# 평균 acc : 0.95

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

print(y_predict)
print(y_test)

# 맹그러바  accuracy_score

accuracy = accuracy_score(y_test, y_predict)
print("accuracy: ", accuracy)



