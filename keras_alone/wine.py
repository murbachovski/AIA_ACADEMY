#test GITHUB AND 복습
#23_03_14 정삭적으로 커밋이 됩니다. 윈도우 좋네...
#정리한 내용들 살펴봅시다.
# <과적합 배제>
# 데이터를 많이 넣는다.
# 노드의 일부를 뺀다. dropout
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, Input, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical #ONE_HOT_ENCODING
from sklearn.preprocessing import RobustScaler #SCALER
import matplotlib.pyplot as plt
import datetime #시간으로 저장해주는 고마운 녀석

date = datetime.datetime.now()
# print(date) #2023-03-14 22:02:28.099457
date = date.strftime('%m%d_%M%M')
filepath = ('./_save/MCP/wine/')
filename = '{epoch:04d}_{val_loss:.4f}_{val_acc:.4f}.hdf5'
#see_val_acc = '{val_acc:.4f}' # save하는 파일에도 val_acc를 심을 수는 없을까?

#1. DATA
path = ('./_data/wine/')
path_save = ('./_save/wine/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(train_csv.shape, test_csv.shape) #(5497, 13) (1000, 12)

# ISNULL
train_csv = train_csv.dropna()
# print(train_csv.shape) #(5497, 13)
# print(train_csv.columns)


# x, y SPLIT
x = train_csv.drop(['quality', 'type'], axis=1)
y = train_csv['quality']
test_csv = test_csv.drop(['type'], axis=1) # 현재 type을 구할 수 없는 실력이다.
# print(x.shape, y.shape) #(5497, 11) (5497,)
'''
print(y)
0       5
1       5
2       5
3       6
4       6
       ..
5492    5
5493    6
5494    7
5495    5
5496    6
'''
#print(np.unique(y,return_counts=True)) #(array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
# ONE_HOT_ENCODING
y = to_categorical(y)
# print(x.shape, y.shape) #(5497, 11) (5497, 10)
'''이런 형태로 변했다.
print(y)
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
'''
#print(np.unique(y,return_counts=True)) #(array([0., 1.], dtype=float32), array([49473,  5497], dtype=int64))

# TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=2222,
    stratify=y #분류 모델이니깐 사용합니다.(균등분배) 균등하게 분배하는지 어떻게 확인할까?
)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (4122, 11) (1375, 11) (4122, 10) (1375, 10)


# SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(x_train.shape, x_test.shape) # (4397, 11) (1100, 11) #하기 전에 train은 늘어났고 test는 줄어들었다. 이유는?
print(y.shape)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_shape=(11,)))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax')) #y_label값을 어떻게 확인하는가?

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=100,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'dacon_wine', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.2, batch_size=200, callbacks=[es, mcp])

# SAVE_MODEL
model.save('./_save/model_dacon_wine/bestOfbest_wine_model.h5')
#model = load_model('_save\MCP\dacon_wine\dacon_wine0315_1010_0213_0.9815_0.6136.hdf5')
#MCPMODEL
#model = load_model('_save\MCP\dacon_wine\(stratifyOFF)dacon_wine0314_4646_0234_1.0691_0.5773.hdf5')
#ACC:  0.59 MCP로 뽑은 모델이 더 좋다. MCP > SAVE model=(stratifyON)dacon_wine0314_2323
#ACC:  0.6272727272727273 #stratify=y주석 = 결과가 많이 좋아졌다. model=(stratifyON)dacon_wine0314_2323
#ACC:  0.5854545454545454 #stratify=y노주석 = 결과가 조금 좋아졌다. model=(stratifyOFF)dacon_wine0314_4646
#ACC:  0.5754545454545454 #stratify=y주석 = 결과가 떨어졌다.  model=(stratifyOFF)dacon_wine0314_4646
#ACC:  0.6054545454545455 #stratify=y노주석 + random_stat변경 = 결과가 많이 좋아졌따. model=(stratifyOFF)dacon_wine0314_4646

#SAVEMODEL
#model = load_model('_save\model_dacon_wine\(stratifyOFF)wine_model.h5')
'''
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=500,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'dacon_wine', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=2000, validation_split=0.3, batch_size=200, callbacks=[es, mcp])
'''
#ACC:  0.5563636363636364 #stratify=y노주석 model=(stratifyON)wine_model
#ACC:  0.5618181818181818 #stratify=y주석 = 결과가 조금 좋아졌다. model=(stratifyON)wine_model
#ACC:  0.58 #stratify=y주석 = 결과가 조금 좋아졌다. model=(stratifyOFF)wine_model 
#ACC:  0.5890909090909091 #stratify=노주석 = 결과가 조금 좋아졌다. model=(stratifyOFF)wine_model 
#ACC:  0.6145454545454545 #stratify=노주석 + random_stat변경 = 결과가 많이 좋아졌따. model=(stratifyOFF)wine_model 

#####결론은 stratify=y 노주석 후 MCP로 모델 뽑은 뒤 stratify=y주석 + random_state변경 후 값이 제일 좋다#####

#4. PREDICT
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc:', results[1])
y_predict = model.predict(x_test)
#print(y_predict)
#print(y_predict.shape)
y_predict_acc = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict_acc)
print('ACC: ', acc)

#5. SUBMIT
test_csv_sc = scaler.transform(test_csv)
y_submit = model.predict(test_csv_sc)
y_submit = np.argmax(y_submit, axis=1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['quality'] = y_submit
submission.to_csv(path_save + 'best_dacon_wine_submit.csv')

#6. PLT
plt.plot(hist.history['acc'], label='acc', color='red')
plt.plot(hist.history['val_acc'], label='val_acc', color='blue')
plt.plot(hist.history['loss'], label='loss', color='green')
plt.plot(hist.history['val_loss'], label='val_loss', color='yellow')
plt.legend()
plt.show()