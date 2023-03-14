import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical # ONE_HOT_ENCODIN
from sklearn.preprocessing import MinMaxScaler, RobustScaler # SCALER

#1. DATA
path = ('./_data/wine/')
path_save = ('./_save/wine/')

x_train = pd.read_csv(path + 'train.csv')
x_test = pd.read_csv(path + 'test.csv')

print(x_train.shape, x_test.shape) #(5497, 14) (1000, 13)
