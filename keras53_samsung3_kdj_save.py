# 현대차(30일) 시가 예측 176,300원~

# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기
# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라.
# 제공된 데이터 외 추가 데이터 사용금지
# 삼성전자 28일(화) 종가 맞추기 (점수배점 0.3)
# 삼성전자 29일(수) 아침 시가 맞추기 (점수배점 0.7) y한칸 건너 뛰어서
#메일 제출
#메일 제목: 김대진 [삼성 2차] 60,350.07원
#첨부 파일: keras53_samsung4_kdj_submit.py
#가중치: _save/samsung/keras53_samsung4_kdj.h5
#오늘밤 23:59분 1차 28일(화) 23시59분 59초 / 29일(수) 23시 59분 59초

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model, save_model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt


# DATA
path = './_data/시험/'
dataset_s = pd.read_csv(path + '삼성전자 주가3.csv', thousands=',', encoding='utf-8')
dataset_h = pd.read_csv(path + '현대자동차2.csv', thousands=',', encoding='utf-8')

# 이상치 적은 값 drop
dataset_s = dataset_s.drop(['전일비', '금액(백만)'], axis=1)
dataset_h = dataset_h.drop(['전일비', '금액(백만)'], axis=1)
# print(dataset_s.head) # 앞 다섯개만 보기
# print(dataset_h.head) # 앞 다섯개만 보기
# print(dataset_s.info())
dataset_s = dataset_s.fillna(0) # filena() 뭐하는 녀석이지?
dataset_h = dataset_h.fillna(0) # 이상치 0으로 맞춰줍니다.
# print(dataset_s.head)
# print(dataset_h.head)

# 액면분할 이후 데이터만 사용
dataset_s = dataset_s.loc[dataset_s['일자'] >= '2018/05/04']
dataset_h = dataset_h.loc[dataset_h['일자'] >= '2018/05/04']

# 오름차순 정렬 #시계열 잘라주기 위해서
dataset_s = dataset_s.sort_values(by=['일자'], axis=0, ascending=True) #현재~과거 데이터 순서를 과거~현재 순서로 바꿔준다.
dataset_h = dataset_h.sort_values(by=['일자'], axis=0, ascending=True)
#print(dataset_s.shape, dataset_h.shape) # (1206, 15) (1206, 15)
# print(dataset_s.head)
# print(dataset_h.head)

# 사용할 cols 지정
#일자, 시가, 고가, 저가, 종가, 전일비, Unnamed: 6, 등락률, 거래량, 금액(백만), 신용비, 개인, 기관, 외인(수량), 외국계, 프로그램, 외인비
feature_cols = ['시가', '고가', '저가', '기관', '거래량', '외국계', '종가']
dataset_s = dataset_s[feature_cols]
dataset_h = dataset_h[feature_cols]

# np.array
dataset_s = np.array(dataset_s)
dataset_h = np.array(dataset_h)

# def 시계열 함수 정의
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i: x_end_number, : -1]
        tmp_y = dataset[x_end_number -1 : y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
SIZE = 3
COLSIZE = 1
x1, y1 = split_xy(dataset_h, SIZE, COLSIZE)
x2, y2 = split_xy(dataset_s, SIZE, COLSIZE)
#print(x1.shape, y1.shape) # (1202, 3, 6) (1202, 3)

# TTS
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1,
    x2,
    y1,
    test_size=0.025,
    shuffle=False,
    random_state=2222
)

# SCALER
scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y_train.shape, y_test.shape)
# (1173, 3, 6) (31, 3, 6)
# (1173, 3, 6) (31, 3, 6)
# (1173, 1) (31, 1)

x1_train = x1_train.reshape(1174*3, 6)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(31*3, 6)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(1174*3, 6)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(31*3, 6)
x2_test = scaler.transform(x2_test)

# Conv1D 넣기 위한 3차원화
x1_train = x1_train.reshape(1174, 3, 6)
x1_test = x1_test.reshape(31, 3, 6)
x2_train = x2_train.reshape(1174, 3, 6)
x2_test = x2_test.reshape(31, 3, 6)


# 모델 구성
input1 = Input(shape=(3, 6))

conv1 = Conv1D(128, 2, activation='relu')(input1)
lstm1 = LSTM(128, activation='relu')(conv1)
dense1 = Dense(128, activation='relu')(lstm1)
drop4 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense2)
dense3 = Dense(64, activation='relu')(drop5)
output1 = Dense(64, activation='relu')(dense3)

# 모델2 구성
input2 = Input(shape=(3, 6))
conv2 = Conv1D(128, 2, activation='relu')(input2)
lstm2 = LSTM(128, activation='relu')(conv2)
drop1 = Dropout(0.3)(lstm2)
dense4 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense5)
output2 = Dense(64, activation='relu')(drop3)

# MERGE
from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(64)(merge1)
merge3 = Dense(32, name='mg3')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])
#model.summary()

# COMPILE
model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(
    monitor='val_loss',
    patience=30,
    mode='min',
    restore_best_weights=True
)
hist = model.fit([x1_train, x2_train], y_train, epochs=10000, batch_size=16, callbacks=[Es], validation_split=0.025)
end_time = time.time()

model.save('./_save/samsung/keras53_samsung4_kdj6.h5')
#model = load_model()
# PREDICT
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
predict = np.round(predict, 2)
print('loss: ', loss, 'predict: ', predict[-1:])
print('걸린 시간: ', end_time - start_time)


#6. PLT
plt.plot(hist.history['loss'], label='loss', color='red')
plt.plot(hist.history['val_loss'], label='val_loss', color='blue')
plt.legend()
plt.show()

# 현대 자동차 30일 시가 예측 : 현재  시가: 175,800원 ~ 177,300 ~ 177,900원
# 176,000원 ~ 178,000원 대로 뽑아보자
# 177,500 ~ 178,000원 대로 뽑아야할 것 같은데 값이 잘 안 나온다.
# loss:  18160996.0 predict:  [[175205.86]] 1
# loss:  11033829.0 predict:  [[176898.55]] 2
# loss:  11000035.0 predict:  [[176128.92]] 3
# loss:  6941394.5 predict:  [[176271.73]] 4
# loss:  16418010.0 predict:  [[175287.95]] 5