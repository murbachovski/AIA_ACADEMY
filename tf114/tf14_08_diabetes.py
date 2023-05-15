import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#1. DATA
x, y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape)
# (442, 10) (442,)