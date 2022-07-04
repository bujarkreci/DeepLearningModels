import VectorCalc as vc

import tensorflow as tf
import sympy as sp

x1 = tf.constant([[4, 2],[1, 3]], dtype=tf.dtypes.float32)
x2 = tf.constant([[1, 5],[6, 0]], dtype=tf.dtypes.float32)
r1, r2, r3, r4 = vc.jacobianTensor(x1,x2)

print('f1-x1:')
for re in r1:
    print(re.numpy())

print('f1-x2:')
for re in r2:
    print(re.numpy())

print('f2-x1:')
for re in r3:
    print(re.numpy())

print('f2-x2:')
for re in r4:
    print(re.numpy())