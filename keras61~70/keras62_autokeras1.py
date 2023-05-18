import autokeras as ak
# from keras.datasets import mnist
import time
import tensorflow as tf


print(ak.__version__)

#1. DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datsets.mnist.load_data()

#2. MODEL
model = ak.ImageClassifier(
    overwrite = True,
    max_trials=2
)


#3. COMPILE, IFT
start = time. time()
model.fit(x_train, y_train, epochs=10, validaion_split=0.2)
end = time.time()

# 최적의 모델 출력
best_model = model.export_model()
print(best_model.summary())

### 최적의 모델 저장
best_model = model.export_model()
path = './autokeras'
best_model.save(path + './KERAS2.h5')

# 평가, 예측
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('BEST: ', results)
