from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split


path = "./_data/ddarung/"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col=0)

print(train_csv.info())
print(train_csv.columns)

x = train_csv.dropna()
x = train_csv.drop(['count'], axis =1)

y = train_csv(['count'])


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)


model = Sequential()


submission['count'] = model.predict(x_test)

submission.to_csv(path + "submit/" + "submit_0904_333.csv")

import pandas as pd

path = "./_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col =0)


train_csv = train_csv.dropna()
x = train_csv.drop(['count'], axis = 1)
y = train_csv(['count'])


x_train, x_test..


y_predict = model.predict(x_test)

submission(['count']) = y_predict

submission.to_csv(path + "submit/" + "subission_092222.csv")