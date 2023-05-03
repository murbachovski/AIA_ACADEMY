import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import time
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime

# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# CALL DATA
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')

# ID COLUMN REMOVE
col = ['ID',
        'Height(Feet)',
        'Height(Remainder_Inches)',
        'Weight(lb)',
        'Weight_Status',
        'Gender',
        'Age'
]

# Weight_Status, Gender => NUMBER
train_df['Weight_Status'] = train_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_df['Gender'] = train_df['Gender'].map({'M': 0, 'F': 1})
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})

train_df = train_df.drop('ID',  axis=1)
test_df = test_df.drop('ID', axis=1)

# PolynomialFeatures DATA 
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(train_df.drop('Calories_Burned', axis=1))
y = train_df['Calories_Burned']

# SCALER
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train, valid SPLIT
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.


#2. MODEL # 대문자 = class
mlp = GridSearchCV(MLPRegressor(),
                    # hidden_layer_sizes=(2000, 600,3),
                    # max_iter=500, # = epochs = 반복 횟수
                    # activation='relu',
                    # solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
                    # random_state=42,
                    # alpha=2.5,
                    verbose=1,
                    refit=True,
                    n_jobs=-1,
                    cv=5
                    )

start_time = time.time()
mlp.fit(X_train, y_train)
end_time = time.time()

# valid PREDICT
y_pred_valid = mlp.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

# test PREDICT
X_test = test_df.values
X_poly_test = poly.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)
y_pred_test = mlp.predict(X_test_scaled)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

# SUBMIT
sample_submission_df['Calories_Burned'] = y_pred_test
sample_submission_df.to_csv(save_path + date + 'submission_MLP_Poly.csv', index=False)


print("최적의 매개변수 : ", mlp.best_estimator_) 

print("최적의 파라미터 : ", mlp.best_params_)

print("최적의 인덱스 : ", mlp.best_index_)

print("BEST SCORE : ", mlp.best_score_)
print("model 스코어 : ", mlp.score(X_poly_test, y_pred_test))

print('걸린 시간: ', round(end_time - start_time, 2),'초')
