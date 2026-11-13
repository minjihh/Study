# m09 00 카피

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
# from sklearn.experimental import enable_halving_search_cv
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y = True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size = 0.8,
    stratify=y,
)

print(x_train.shape)  # (1437)



n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 333)

parameters = [
    {'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12],
    'learning_rate' : [0.1, 0.01, 0.001]}, # 18
    {'max_depth' : [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, # 12
    {'min_child_weight' :[2,3,4,10], 'learning_rate' : [0.1, 0.01, 0.001]}  # 12
]  # 42


#2. 모델
xgb = XGBClassifier()
model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
             verbose = 1,
            #  n_jobs = -1, # 모든 cpu 다 쓰겠다.
            #  random_state = 333,   # GridSearch에는 random_state없음
             refit= True,
            #  n_iter = 10,  # default = 10
            min_resources = 100,  # 5 * 2 * 10 = 2*cv*클래스 갯수 // 첫번째 훈련에서 min_resources 만큼 사용하겠다.
            max_resources = 1437,
            factor = 2,
             )

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)

#4. 평가, 예측
print("==========================================")
print('model.score :', model.score(x_test, y_test))
print('best_score :', model.best_score_) # train 데이터로 테스트한것 중 best -> 과적합되있음

print("걸린시간 :", round(end - start, 2), "초")


# n_candidates : parameters // 매 iter마다 best가 뽑힘 // iter 마다 /factor만큼 줄어듬
# n_resources : The minimum amount of resource that any candidate is allowed to use for a given iteration. // 이전 iter의 *factor 만큼 매 iter마다 늘어남