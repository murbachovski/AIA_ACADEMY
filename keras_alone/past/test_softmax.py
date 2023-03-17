import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)
print(type(x))

#1-2. ONE_HOT_ENCODING
from keras.utils import to_categorical
y = to_categorical(y)

#1-3. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=333,
)
print(np.unique(y_train, return_counts=True))

#2. MODEL
model = Sequential()
model.add(Dense(10, input_dim=x.shape[1]))
model.add(Dense(5, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

