import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error,r2_score
# / // \ \\ 전부 같다. 예외) 텍스트에서 \하면 줄 바꿈 예약이됨. \\ 두개는 가능.
#1-1경로 생성.
path = './_data/spark/'
path_save = './save/spark'
train_files = glob.glob(path + "TRAIN/*.csv") #이 폴더에 있는 모든 파일들을 txt화시킴
#print(train_files)
test_input_files = glob.glob(path + "test_input/*.csv") #경로에서는 대소문자 상관 없음.
#print(test_input_files)
#리스트형태로 나오기 때문에 for문으로 부름

###################################train폴더##########################################
li = [] #리스트 값 저장.
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header= 0, #-> header= 0맨위에 행 제거
                     encoding='utf-8-sig') #'utf-8-sig'가 한글로 인코딩 해주는거
    
    li.append(df)
    
# print(li) #[35064 rows x 4 columns] * 17개
# print(len(li)) #17개

train_dataset = pd.concat(li, axis=0,
                          ignore_index=True) #인덱스 제거
#print(train_dataset) #[596088 rows x 4 columns]

#####################################test폴더###########################################
li = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header= 0,
                     encoding='utf-8-sig')
    
    li.append(df)
    
#print(li) #[7728 rows x 4 columns] * 17개
# print(len(li)) #17개

test_dataset = pd.concat(li, axis=0,
                          ignore_index=True) #인덱스 제거
#print(test_dataset) #[131376 rows x 4 columns]

###################################측정소 라벨인코더##########################################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_dataset['locate'] = le.transform(test_dataset['측정소']) #fit트랜스폼을 쓰면 안됨. 위에 train값에 들어가는 번호가 그대로 들어가야되기 때문

#print(train_dataset) #[596088 rows x 5 columns]
#print(test_dataset) #[131376 rows x 5 columns]
#print("=======================드랍후=============================")
train_dataset = train_dataset.drop(['측정소'], axis = 1)
test_dataset = test_dataset.drop(['측정소'], axis = 1)
#print(train_dataset) #[596088 rows x 4 columns]
#print(test_dataset) #[131376 rows x 4 columns]

################################일시 -> 월,일 시간으로 분리 ##################################

#12-31 21 : 00 -> 12와 21일 추출
#print(train_dataset.info()) #object -> string
train_dataset['Month'] = train_dataset['일시'].str[:2]  #month라는 컬럼을 생성
#print(train_dataset['Month'])
train_dataset['hour'] = train_dataset['일시'].str[6:8]
#print(train_dataset['hour'])
#print(train_dataset) #[596088 rows x 6 columns]

test_dataset['Month'] = test_dataset['일시'].str[:2]
test_dataset['hour'] = test_dataset['일시'].str[6:8]

#일시 드랍.
train_dataset = train_dataset.drop(['일시'], axis = 1)
test_dataset = test_dataset.drop(['일시'], axis = 1)
#print(train_dataset) #[596088 rows x 5 columns]

###str -> int
#train_dataset['Month'] = train_dataset['Month'].astype('int32')
#train_dataset['Month'] = pd.to_numeric(train_dataset['Month'])
train_dataset['Month'] = pd.to_numeric(train_dataset['Month']).astype('int16') #인트를 마음대로 조절할수있다는 장점이 있음
train_dataset['hour'] = pd.to_numeric(train_dataset['hour']).astype('int16')
#즉, 메모리가 줄어들게 할 수있음.

test_dataset['Month'] = pd.to_numeric(test_dataset['Month']).astype('int16') #인트를 마음대로 조절할수있다는 장점이 있음
test_dataset['hour'] = pd.to_numeric(test_dataset['hour']).astype('int16')

# print(train_dataset.info())
# Data columns (total 5 columns):
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   연도      596088 non-null  int64
#  1   PM2.5   580546 non-null  float64
#  2   locate  596088 non-null  int32
#  3   Month   596088 non-null  int16
#  4   hour    596088 non-null  object
# dtypes: float64(1), int16(1), int32(1), int64(1), object(1)
# memory usage: 17.1+ MB
# None

# print(test_dataset.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 131376 entries, 0 to 131375
# Data columns (total 5 columns):
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   연도      131376 non-null  int64
#  1   PM2.5   53040 non-null   float64
#  2   locate  131376 non-null  int32
#  3   Month   131376 non-null  int16
#  4   hour    131376 non-null  int16
# dtypes: float64(1), int16(2), int32(1), int64(1)
# memory usage: 3.0 MB
# None

#4500쯤에 결측치가 군집 존재. 전체는 15000개 예상
# 이쯤이 미세먼지 많은곳으로 추정. 
# 모델을 돌려서y값으로 두어서 측정해야됨.

#####################################결측치 제거#######################
#596088 -> 580546
train_dataset = train_dataset.dropna()
# print(train_dataset.info()) 580546 

# 파생 feature -> 주말, 평일 이런거를 다른 feature를 예측. 여기 데이터는 계절으로 파생feature

y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'], axis = 1)
#print(x, '\n', y )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, train_size= 0.75, random_state=369
) #셔플을 true로 준이유는 날짜가 셔플되는게 아님.

parameter = {'n_estimators' :1,  
              'learning_rate' : 0.3,
              'max_depth': 3,        
              'gamma': 0,
              'min_child_weight': 1, 
              'subsample': 0.5,
              'colsample_bytree': 1,
              'colsample_bylevel': 1., 
              'colsample_bynode': 1,
              'reg_lambda': 1,
              'random_state': 369,
              'n_job' : -1
}

#2. 모델
model = XGBRegressor()

#3. 컴파일, 훈련
model.set_params(
    **parameter,
    eval_metric='mae',
    early_stopping_rounds=200,
) #-> keras컴파일
start = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 1
          )
end = time.time()
print("걸린 시간 : ", round(end -start, 2), "초")

#4.평가, 예측

y_predict = model.predict(x_test)

results = model.score(x_test, y_test)
print("model_score :", results)

r2 = r2_score(y_test,y_predict)
print("r2스코어 :", r2)

mae = mean_absolute_error(y_test,y_predict)
print("mae :", mae)