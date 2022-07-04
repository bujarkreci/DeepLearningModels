import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
print(x)

x = tf.Variable(x)

# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
print(y)

x_grad = t.gradient(y, x)
print(x_grad)

#The gradient of the function y=2*xT*x  with respect to x should be 4*x.
print( x_grad == 4 * x)

#Now let us calculate another function of x.
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
print(t.gradient(y, x)) # Overwritten by the newly calculated gradient

#backward computation
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`

# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
print(x_grad == u)

print(t.gradient(y, x) == 2 * x)

def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c

a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
print(d_grad)

print(d_grad == d / a)



