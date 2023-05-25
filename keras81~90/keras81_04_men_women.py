import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score

# Data paths
train_data_dir = 'd:/study/study_data/'
validation_data_dir ='d:/study/study_data/'

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Adjust the target size according to your requirements
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),  # Adjust the target size according to your requirements
    batch_size=32,
    class_mode='binary'
)

# Load VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Freeze base VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Predict on new images
new_image_path = 'C:/Users/murbachovski/Desktop/STUDY/kim.jpg'  # Replace with the path to your new image
new_image = tf.keras.preprocessing.image.load_img(new_image_path, target_size=(224, 224))
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = new_image / 255.0
new_image = tf.expand_dims(new_image, axis=0)

prediction = model.predict(new_image)
if prediction[0] >= 0.5:
    print("남자")
else:
    print("여자")
# 250/250 [==============================] - 131s 501ms/step - loss: 0.3439 - accuracy: 0.8543 - val_loss: 0.1537 - val_accuracy: 0.9376
# 1/1 [==============================] - 1s 965ms/step
# 남자