#Learning: https://www.tensorflow.org/guide/autodiff
import tensorflow as tf
import sympy as sp

def gradientTensor(x1,x2,):
    with tf.GradientTape(persistent=True) as gfg:
        gfg.watch([x1,x2])

        f1= x1**3+x2**2
        f2= 2*x1**2+4*x2

    res1,res2 = gfg.gradient(f1, [x1,x2]) 
    res3, res4 = gfg.gradient(f2, [x1,x2])
    sumi = gfg.gradient({'f1': f1, 'f2': f2}, x1).numpy()

    return res1, res2, res3, res4, sumi

def jacobianTensor(x1, x2):
    with tf.GradientTape(persistent=True) as gfg:
        gfg.watch([x1,x2])

        f1= x1**3+x2**2
        f2= 2*x1**2+4*x2

    res1, res2  = gfg.jacobian(f1, [x1,x2])
    res3, res4  = gfg.jacobian(f2, [x1,x2])
    return res1, res2, res3, res4