import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-3.0, 3.0, 100)

with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)
  y = tf.nn.relu(2*x+0)
  #y = 1/(1+tf.exp(-x))  

dy_dx = tape.gradient(y, x)

plt.plot(x, y, label='y')
#plt.plot(x, dy_dx, label='dy/dx')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()