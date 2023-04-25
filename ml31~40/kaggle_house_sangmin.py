from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_house/'


train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype =='object':
        train_csv[i] = le.fit_transform(train_csv[i])
    for u in test_csv.columns:
        if test_csv[u].dtype =='object':
            test_csv[u] = le.fit_transform(test_csv[u])
            
# 1.3 결측지 확인

# 1.7 Scaler
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
x = train_csv.drop(['SalePrice'], axis=1)
y = train_csv['SalePrice']
bogan = [x.fillna(0),
        train_csv.fillna(method='ffill'),
        train_csv.fillna(method='bfill'),
        train_csv.fillna(value = 77)
        ]
bogan_test = [test_csv.fillna(0),
            test_csv.fillna(method='ffill'),
            test_csv.fillna(method='bfill'),
            test_csv.fillna(value = 77)
            ]

print(x.shape)
print(test_csv.shape)
for j in bogan:
    x = j
    for k in bogan_test:
        test_csv = k
        print(x.shape)
        print(test_csv.shape)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True)
        # 2. 모델구성
        model = RandomForestRegressor(random_state=123)

        #3. 훈련
        model.fit(x_train,y_train)

        # 4. 평가, 예측
        loss = model.score(x_test, y_test)
        print('loss : ', loss)

        y_predict = model.predict(x_test)

        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_predict)
        print('r2 : ', r2)

            # #4. 평가,예측
        submittion = model.predict(test_csv)
        submission['SalePrice'] = submission
        submission.to_csv('./_save/kaggle_house/submit.csv')