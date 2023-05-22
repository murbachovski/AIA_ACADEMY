import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28).astype('float32') / 255.0

# 2. MODEL
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=64, node3=64, lr=0.001):
    model = Sequential()
    model.add(LSTM(node1, activation=activation, input_shape=(28, 28)))
    model.add(Dropout(drop))
    model.add(Dense(node2, activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(node3, activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')

    return model

def create_hyperparameters():
    batch_sizes = [100, 200, 300, 400, 500, 600, 700]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.9]
    
    return {
        'batch_size': batch_sizes,
        'optimizer': optimizers,
        'drop': dropout_rates,
        'activation': activations,
        'lr': learning_rates
    }

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

keras_model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import RandomizedSearchCV

model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

# Define callbacks
callbacks = [
    EarlyStopping(patience=3, monitor='val_loss'),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', verbose=1)
]

model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=callbacks)

print("Best parameters:", model.best_params_)
print("Best score:", model.best_score_)
print("Test accuracy:", model.score(x_test, y_test))

# Best parameters: {'optimizer': 'adam', 'lr': 0.001, 'drop': 0.1, 'batch_size': 100, 'activation': 'elu'}
# Best score: 0.9703333377838135
# 100/100 [==============================] - 2s 20ms/step - loss: 0.0627 - acc: 0.9809
# Test accuracy: 0.98089998960495