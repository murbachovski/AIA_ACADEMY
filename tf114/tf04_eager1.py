import tensorflow as tf
print(tf.__version__)


tf.compat.v1.disable_eager_execution() # 즉시 실행모드 Off # tf2.0 -> 1.0방식으로
print(tf.executing_eagerly()) # False # 즉시 실행모드.

tf.compat.v1.enable_eager_execution() # 즉시 실행모드 On # tf1.0 -> 2.0방식으로
print(tf.executing_eagerly()) # true # 즉시 실행모드.

aaa = tf.constant('hello')
sess = tf.compat.v1.Session()
print(sess.run(aaa))
print(aaa)  # tf2에서는 출력이 가능하다.
