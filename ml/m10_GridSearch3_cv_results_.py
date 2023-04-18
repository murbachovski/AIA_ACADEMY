# 그물망처럼 찾겠다.
# 파라미터 전체를 다 하겠다. / 모델 정의 부분이나 모델 훈련 부분에 있음
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, #stratify=y # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'degree': [3, 4, 5]}, # 12
    {'C': [1, 10, 100], 'kernel': ['rbf, lienear'], 'gamma': [0.001, 0.0001]}, # 12
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.01, 0.001, 0.0001], 'degree': [3, 4]}, # 24
    {'C':[0.1, 1], 'gamma': [1,10]} # 4
]   # 총 52번 돌아갑니다.

#2. MODEL # 대문자 = class
model = GridSearchCV(SVC(), parameters,
                     cv=5,
                     verbose=1,
                     refit=True,    # default, 최상의 파라미터로 출력
                    #  refit=False, # 최종 파라미터로 출력
                     n_jobs=-1)
                                # GridSearchCV default = StratifiedKFold
#3. COMPILE
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_) 

print("최적의 파라미터 : ", model.best_params_)

print("최적의 인덱스 : ", model.best_index_)

print("BEST SCORE : ", model.best_score_)
print("model 스코어 : ", model.score(x_test, y_test))


y_predict = model.predict(x_test)
print('ACC: ', accuracy_score(y_test, y_predict))
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# 최적의 인덱스 :  0
# BEST SCORE :  0.9916666666666668
# model 스코어 :  1.0
# ACC:  1.0

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠ACC: ', accuracy_score(y_test, y_pred_best))
# 최적 튠ACC:  1.0

print('걸린 시간: ', round(end_time - start_time, 2),'초')

###############################################################################
print('==============================================================================================================================================')
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=False)) # 오름차순이 = True = 디폴트다.

print(pd.DataFrame(model.cv_results_).columns)
# [46 rows x 17 columns]
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
#        'split0_test_score', 'split1_test_score', 'split2_test_score',
#        'split3_test_score', 'split4_test_score', 'mean_test_score',
#        'std_test_score', 'rank_test_score'],
#       dtype='object')

path = './temp/' # temp 만들어줘야 save 됩니다.
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path + 'm10_GRSC3.csv')

