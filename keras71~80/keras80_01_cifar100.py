import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.applications import EfficientNetB0

# CIFAR-100 데이터 세트 로드
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 이미지 크기 조정 (배치 사이즈로 분할)
batch_size = 1000
num_batches = len(x_train) // batch_size

resized_images_train = []
for i in range(num_batches):
    batch = x_train[i * batch_size : (i + 1) * batch_size]
    resized_images_train.append(tf.image.resize(batch, (64, 64)))
x_train = tf.concat(resized_images_train, axis=0)

resized_images_test = tf.image.resize(x_test, (64, 64))
x_test = preprocess_input(resized_images_test)

# 모델 로드
model = EfficientNetB0(weights='imagenet')

# 이미지 분류 수행
predictions = model.predict(x_test)
top_predictions = decode_predictions(predictions, top=1)

# 결과 출력
for i in range(len(top_predictions)):
    print(f"실제 레이블: {y_test[i]}, 예측 레이블: {top_predictions[i][0][1]}")
