# 잡음 제거
# 기미, 주근깨 제거
# 이미지 데이터에 많이 씀
# 하지만 모든 데이터에 사용 가능
# 준지도 학습
# x로 x를 훈련시킨다. (y값은 필요가 없다)
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

