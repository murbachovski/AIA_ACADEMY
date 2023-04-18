import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

#1. DATA
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0] # 같으면 0 다르면 1

#2. MODEL
model = SVC()

#3. COMPILE
model.fit(x_data, y_data)

#4. PREDICT
y_predict = model.predict(x_data)

results = model.score(x_data, y_data)
print('model.score: ', results)

acc = accuracy_score(y_data, y_predict)
print('acc_score:', acc)

# model.score:  1.0
# acc_score: 1.0