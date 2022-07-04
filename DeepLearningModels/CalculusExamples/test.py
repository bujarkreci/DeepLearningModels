import tensorflow as tf

a = tf.Variable(initial_value=tf.zeros(shape=(3,)))

print ('a = ', a)
print('a[-1]', a[-1])