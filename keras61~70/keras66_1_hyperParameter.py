import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout


#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')/255

#2. MODEL
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(512, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(512, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy'
                  )
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizer = ['adam',  'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'linear']
    return {'batch_size' : batchs,
            'optimizer' : optimizer,
            'dropout' : dropout,
            'activation' : activation
            }

hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import GridSearchCV
model1 = build_model()
model = GridSearchCV(model1, hyperparameters, cv=3)
model.fit(x_train, y_train, epochs=3)