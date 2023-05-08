import tensorflow as tf
print(tf.__version__)

print(tf.executing_eagerly()) # False # 즉시 실행모드.

aaa = tf.constant('hello')
sess = tf.compat.v1.Session()
print(sess.run(aaa))
print(aaa)  # tf2에서는 출력이 가능하다.
