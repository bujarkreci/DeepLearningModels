import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
class Example:
    def __init__(self, x1, x2, target, lr=0.01):
        self.x1 = tf.Variable(x1,dtype=tf.float32)
        self.x2 = tf.Variable(x2,dtype=tf.float32)
        self.target = tf.Variable(target,dtype=tf.float32)
        self.lr = lr
        self.variables = [self.x1, self.x2]

    @tf.function
    def iterate(self):
        with tf.GradientTape() as tape:
            loss = (self.target - self.x1 * self.x2)**2
        #it is rather dangerous to use self.gradients here
        grads = tape.gradient(loss, self.variables)        
        #print(grads)
        for g, v in zip(grads, self.variables):
            print(g,'---',v)
            v.assign_add(-self.lr * g)
            
    @tf.function
    def compute_update(self):
        with tf.GradientTape() as tape:
            loss = (self.target - self.x1 * self.x2)**2
        #return a list of gradients
        return tape.gradient(loss, self.variables)

    @tf.function
    def apply_update(self, grad): #receive the gradients as arguments
        for g, v in zip(grad, self.variables):
            v.assign_add(-self.lr * g)

example = Example(1, 3, 5)

example.iterate()
#print(example.variables)
#example.apply_update(example.compute_update())
#print(example.variables)
