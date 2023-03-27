# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기
# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라.
# 제공된 데이터 외 추가 데이터 사용금지
# 삼성전자 28일(화) 종가 맞추기 (점수배점 0.3)
# 삼성전자 29일(수) 아침 시가 맞추기 (점수배점 0.7) y한칸 건너 뛰어서
#메일 제출
#메일 제목: 김대진 [삼성 1차] 60,350.07원
#메일 제목: 김대진 [삼성 2차] 60,350.07원
#첨부 파일: keras53_samsung2_kdj_submit.py
#첨부 파일: keras54_samsung4_kdj_submit.py
#가중치: _save/samsung/keras53_samsung2_kdj.h5/hdf5
#가중치: _save/samsung/keras53_samsung4_kdj.h5/hdf5
#오늘밤 23:59분 1차 23일(월) 23시59분 59초 / 28일(화) 23시 59분 59초

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional


# DATA
path = ('./_data/시험/')
path_save = ('./_save/samsung/')

x1 = pd.read_csv(path + '삼성전자주가.csv', index_col=0)
x2 = pd.read_csv(path + '현대자동차.csv', index_col=0)
# print(x1, x2)
# print(x1.shape, x2.shape) # (3260, 17) (3140, 17)

# ISNULL
x1 = x1.dropna()
x2 = x2.dropna()
# print(x1, x2)
# print(x1.shape, x2.shape) # (3257, 17) (3140, 17) #결측치가 있구나.
x1 = x1[0 : 3140]
x2 = x2[0 : 3140]
# print(x1.shape) # (3140, 17) # x1, x2 x값을 맞춰주었다.

# x, y SPLIT
x1 = x1.drop([x1.columns[4]], axis=1)
x2 = x2.drop([x2.columns[4]], axis=1)
y = x1[x1.columns[4]]

# x1 = x1.astype(np.float32)
# x2 = x2.astype(np.float32)
# y = y.astype(np.float32)

# print(x1, x2)
# print(x1.shape, x2.shape, y.shape) # (3140, 16) (3140, 16) (3140,)

######################################################### 등락률에 대해서 라벨인코더가 필요한 상황
######################################################### 삼성과 현대의 x 크기가 다르다. 맞춰주어야 할거 같은데?
######################################################### 일자도 라벨인코더 해주면 좋을거 같은데? (SCALER가 오류 나네)

# le = LabelEncoder() # 정의 핏 트랜스폼
# le.fit(train_csv['type'])
# aaa = le.transform(train_csv['type'])
# train_csv['type'] = aaa
# test_csv['type'] = le.transform(test_csv['type'])

# print(train_csv.shape, test_csv.shape) #(5497, 13) (1000, 12)
# print(train_csv.columns)

# print(x1)
# print(type(x1)) 
# print(x1.shape) 
# print(np.unique(x1, return_counts=True))
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(x2)
# print(type(x2)) 
# print(x2.shape) 
# print(np.unique(x2, return_counts=True))

# TRAIN_TEST_SPLIT
from sklearn.model_selection import train_test_split
x1_train, x1_test = train_test_split(
    x1,
    train_size=0.7,
    random_state=2222
)
x2_train, x2_test = train_test_split(
    x2,
    train_size=0.7,
    random_state=2222
)
y_train, y_test = train_test_split(
    y,
    train_size=0.7,
    random_state=2222
)

# #SACLER
# scaler = RobustScaler()
# scaler.fit(x1_train)
# x_train = scaler.transform(x1_train)
# x_test = scaler.transform(x1_test)

print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape) # (2198, 16) (942, 16) (2198, 16) (942, 16) (2198,) (942,)

# MODEL
input1 = Input(shape=(17,))
Dense1 = Dense(256, activation='relu', name='stock1')(input1)
Dense2 = Dense(128, activation='relu', name='stock2')(Dense1)
Dense3 = Dense(64, activation='relu', name='stock3')(Dense2)
output1 = Dense(32, activation='relu', name='output1')(Dense3)

# MODEL2
input2 = Input(shape=(17,))
Dense11 = Dense(256, name='weather1', activation='relu')(input2)
Dense12 = Dense(128, name='weather2', activation='relu')(Dense11)
Dense13 = Dense(64, name='weather3')(Dense12)
Dense14 = Dense(32, name='weather4')(Dense13)
output2 = Dense(16, name='output2')(Dense14)

# MERGE
from tensorflow.keras.layers import concatenate, Concatenate # (소문자)함수, (대문자)클래스
merge1 = concatenate([output1, output2], name='mg1') #리스트 형태로 받아들임.
merge2 = Dense(32, activation='relu', name='mg2')(merge1)
merge3 = Dense(16, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
x_train = [x1_train, x2_train]
model.fit(x_train, y_train, epochs=10, batch_size=10)

#4. PREDICT
x_test = [x1_test, x2_test]
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss)