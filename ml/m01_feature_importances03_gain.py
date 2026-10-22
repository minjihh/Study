from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


#1. 데이터 

x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)  # (150, 4) (150,)

random_state = 124

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = random_state,
    stratify = y   # 회귀에서는 쓰면 안됨
)
 
#2. 모델구성
model = XGBClassifier(random_state = random_state)


#3. 컴파일, 훈련
model.fit(x_train, y_train)
print("===============", model.__class__.__name__, "=================")

#4. 평가, 예측
score = model.score(x_test, y_test)
print("acc: ", score)   # 회귀에서 model.score는 R2score
print("F 스코어 :", model.feature_importances_) 
# F스코어 : 빈도수 
gain_score = model.get_booster().get_score(importance_type = 'gain')
print("게인 스코어 :", gain_score) # 높을수록 좋음, 손실함수의 개선율을 나타냄 (실질적인 성능에 기여)
# Gain 스코어 : 손실함수의 개선 점수. 높으면 갯너 많이 됐다!



