import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score


# Load train and test data
path='./_data/ai_factory/'
save_path= './_save/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])


# Select subset of features for Autoencoder model
features = ['air_inflow','air_end_temp','out_pressure','motor_current','motor_rpm','motor_temp','motor_vibe']

# Split data into train and validation sets
x_train, x_val = train_test_split(data[features], train_size=0.8, random_state=2222)

# Normalize data
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# Reshape input data for LSTM layer
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])

# Define Autoencoder model with LSTM layer
input_layer = Input(shape=(1, len(features)))
lstm_layer1 = LSTM(128, activation='relu', return_sequences=True)(input_layer)
dropout1 = Dropout(0.4)(lstm_layer1)
lstm_layer2 = LSTM(64, activation='relu')(dropout1)
dropout2 = Dropout(0.4)(lstm_layer2)
dense_layer1 = Dense(32, activation='relu')(dropout2)
dropout3 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(16, activation='relu')(dropout3)
decoder = Dense(len(features), activation='sigmoid')(dense_layer2)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile Autoencoder model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train Autoencoder model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
autoencoder.fit(x_train.reshape(-1, 1, len(features)), x_train.reshape(-1, 1, len(features)), epochs=10, batch_size=64, validation_data=(x_val.reshape(-1, 1, len(features)), x_val.reshape(-1, 1, len(features))), callbacks=[es])


# Predict anomalies in test data

test_data = scaler.transform(test_data[features])
test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
predictions = autoencoder.predict(test_data)
mse = ((test_data - predictions) ** 2).mean(axis=1)
threshold = mse.mean() + mse.std() * 2  # Set threshold based on mean and standard deviation of MSE

# # Evaluate model performance
binary_predictions = [1 if x > threshold else 0 for x in mse]
# acc = accuracy_score(test_data['type'], binary_predictions)
# print('Accuracy:', acc)

# f1_score = f1_score(test_data['type'], binary_predictions, average='macro')
# print('F1 Score:', f1_score)

# Save predictions to submission
submission['label'] = pd.DataFrame({'Prediction': binary_predictions})

#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)
