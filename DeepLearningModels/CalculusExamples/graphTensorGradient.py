import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(0.0, 10.0, 200+1)

with tf.GradientTape() as tape:
  tape.watch(x)
  y = 0+0.64*x

dy_dx = tape.gradient(y, x)
plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()