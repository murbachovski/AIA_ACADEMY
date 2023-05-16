import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # 즉시모드 안해 1.0
# tf.compat.v1.enable_eager_execution()   # 즉시모드 해 2.0

print(tf.__version__)
print(tf.executing_eagerly())
# 1.14.0
# True

gpus =  tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as a:
        print(a)
else:
    print('NO GPU')

# 2.7.4
# True
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')