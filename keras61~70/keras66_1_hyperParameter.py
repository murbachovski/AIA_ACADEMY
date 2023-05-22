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
    batchs = [100, 200, 300, 400, 500, 600, 700]
    optimizer = ['adam',  'rmsprop', 'adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'linear']
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.9]
    return {'batch_size' : batchs,
            'optimizer' : optimizer,
            'drop' : dropout,
            'activation' : activation,
            'lr' : learning_rate
            }

hyperparameters = create_hyperparameter()
print(hyperparameters)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# model1 = build_model()
# model = GridSearchCV(keras_model, hyperparameters, cv=3)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=4, n_iter=2, verbose=1)
import time
start = time.time()
model.fit(x_train, y_train, epochs=3)
end = time.time()

print('걸린시간',end - start, 2)
print("최상의 파라미터", model.best_params_)
print('model.best_estimater_', model.best_estimator_)
print('model.best_socre', model.best_score_)
print('model.socre', model.score(x_test, y_test))
# 걸린시간 10.706912279129028 2
# 최상의 파라미터 {'optimizer': 'adadelta', 'drop': 0.5, 'batch_size': 400, 'activation': 'elu'}
# model.best_estimater_ <keras.wrappers.scikit_learn.KerasClassifier object at 0x0000027668624DC0>
# model.best_socre 0.13316666334867477 Train에 대한
# model.socre 0.9606000185012817 Test에 대한

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc:', accuracy_score(y_test, y_predict))


# 튜닝후
# 걸린시간 39.18590807914734 2
# 최상의 파라미터 {'optimizer': 'rmsprop', 'lr': 0.001, 'drop': 0.5, 'batch_size': 700, 'activation': 'relu'}
# model.best_estimater_ <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002C40B6B9940>
# model.best_socre 0.95414999127388
# 15/15 [==============================] - 0s 5ms/step - loss: 0.1211 - acc: 0.9649
# model.socre 0.964900016784668
# 313/313 [==============================] - 1s 2ms/step
# acc: 0.9649