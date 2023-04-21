import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import joblib
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping as es

# Load train and test data
path='./_data/ai_factory/'
save_path= './_save/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# 
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])
# print(train_data.columns)

# 
features_x = ['air_inflow']
features_y = ['out_pressure']

print(train_data)
print(train_data.shape) # (2463, 8)

x = train_data[features_x]
y = test_data[features_y]
print(x.shape, y.shape) # (2463, 1) (7389, 1)
y = y[:2463]

#
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=333,
    shuffle=True,
    train_size=0.8
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# 2. model build
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(512,activation=LeakyReLU(0.15)))
model.add(Dropout(1/16))
model.add(Dense(512,activation=LeakyReLU(0.15)))
model.add(Dropout(1/16))
model.add(Dense(512,activation=LeakyReLU(0.15)))
model.add(Dropout(1/16))
model.add(Dense(512,activation=LeakyReLU(0.15)))
model.add(Dropout(1/16))
model.add(Dense(512,activation=LeakyReLU(0.15)))
model.add(Dropout(1/16))
model.add(Dense(1,activation='sigmoid'))

# 3. compile,training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size= 200 #len(x_train)//99
          ,callbacks=es(monitor='val_loss',mode='min',patience=20,verbose=True,restore_best_weights=True))

# 4. predict,save
print(x_train.shape)
y_predict = model.predict(x_test)
submission['label'] = y_predict
sub = submission['label']

import datetime
now=datetime.datetime.now().strftime('%m월%d일%h시%M분')
print(sub.value_counts())
sub.to_csv(f'{save_path}{now}_submission.csv',index=False)
