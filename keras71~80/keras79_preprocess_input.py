# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np

# model = ResNet50(weights='imagenet')
# # model = ResNet50(weights=None)
# # model = ResNet50(weights='경로')

# path = './suit.PNG'

# img = image.load_img(path, target_size=(224, 224))
# # print(img) # <PIL.Image.Image image mode=RGB size=224x224 at 0x1A4B2810550>

# x = image.img_to_array(img)
# print('================================================= image.img_to_array(img)')
# # print(x, '\n', x.shape)
# # print(np.min(x), np.max(x))

# x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
# # print(x.shape) (1, 224, 224, 3)

# x = np.expand_dims(x, axis=0)
# # print(x.shape) (1, 224, 224, 3)

# ####################### -1 에서 1 사이로 정규화 ##################
# x = preprocess_input(x)
# # print(np.min(x), np.max(x)) # -123.68 151.061

# print('=====================================================')
# x_pred = model.predict(x)
# print(x_pred, '\n', x_pred.shape)

# print('결과는 : ', decode_predictions(x_pred, top=5))

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
path = './suit.PNG'

img = image.load_img(path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape)
print('결과는:', decode_predictions(x_pred, top=5))
