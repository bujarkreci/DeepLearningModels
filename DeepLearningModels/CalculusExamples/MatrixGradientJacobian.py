# Importing the library
import tensorflow as tf
import sympy as sp

#for calculating jacobian  
#x1 = tf.constant([[4, 2],[1, 3]], dtype=tf.dtypes.float32)
#x2 = tf.constant([[1, 5],[6, 0]], dtype=tf.dtypes.float32)

x1 = tf.constant([[4, 2],[1, 3]], dtype=tf.dtypes.float32)
x2 = tf.constant([[1, 5],[6, 0]], dtype=tf.dtypes.float32)

#for calculating gradient
#x1v=tf.Variable(5,dtype=tf.float32)
#x2v=tf.Variable(2,dtype=tf.float32)
#f1= x1v**3+x2v**2
#f2= 2*x1v**2+4*x2v
# Using GradientTape
with tf.GradientTape(persistent=True) as gfg:
  gfg.watch([x1,x2])
  #gfg.watch(x2)
  #f = x1**3+x2**2
  f1= x1**3+x2**2
  f2= 2*x1**2+4*x2

  
# Computing jacobian
#res  = gfg.jacobian(f, [x1,x2]) 
res1 = gfg.gradient(f1, [x1,x2]) 
res2 = gfg.gradient(f2, [x1,x2])
for re in res1:
    print(re.numpy())

for re in res2:
    print(re.numpy())

print()
print(gfg.gradient({'f1': f1, 'f2': f2}, x1).numpy())

# Printing result
#print("res: ",res) #result will be 27
#print("res gradient: ",res1) #result will be 27

#------------------------Another---------------------


