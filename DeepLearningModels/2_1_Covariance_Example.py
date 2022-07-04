import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#this approach have 2 point clouds with the same shape but different positioons and different mean values.
# 1000 random 2D points cov=[[1, 0.5],[0.5, 1]] oval like points oriented from bootom_left to top_right
num_samples = 3
#negative_samples = np.random.multivariate_normal(mean=[0,3], cov = [[1, 0.5],[0.5, 1]], size = num_samples)
#positive_samples = np.random.multivariate_normal(mean=[3,0], cov = [[1, 0.5],[0.5, 1]], size = num_samples)
negative_samples = np.array([[-1,0],[2,1],[3,3]], dtype="float32")
positive_samples = np.array([[-2,1],[-3,0],[1,1]], dtype="float32")

# shape will be (5, 2)
print('negative samples= ', negative_samples)
print('positiva samples= ', positive_samples)

#now let them stack into a single array (2000, 2)
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

print('inputs shape= ', inputs.shape)
print('inputs= ', inputs)

#now will generate corresponding target labels, an array of 0s and 1s of shape (2000, 1 where targets[i, 0] = 0 if inputs[i] 
#belong to class 0 which means target[i, 0]<0.5) and (targets[i, 0 ] = 1 if inputs[i] belong to class 1 which mean target[i, 0]>0.5)
a = np.ones((num_samples,1), dtype = "float32")
q = np.zeros((num_samples, 1), dtype = "float32")
u = 10.0 * a
targets = np.vstack((q, u)).astype(np.float32)
print('targets= ', targets)

#now let's look the graphic of the result with matplot
plt.scatter(inputs[:,0], inputs[:,1], c=targets[:, 0])
plt.show()

#now let's create a linear classifier that can learn to separate these 2 blobs
#linear classifier is an affine transformation (prediction = W*inputs + b) that will minimize the square of the difference
#between prediction and target

#input will be 2D point and output prediction (score) per every sample each
#output prediction will be a single score per sample (close to 0 if the sample is predicted to be in class 0) and
#(close to 1 if the sample is predicted to be in class 1)
input_dim = 2
output_dim =1
W = tf.Variable(initial_value=np.array([[3],[2]]), dtype="float32")
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

print('W initial start=', W)
print('B initial start=', b) 


def model(inputs):
    a = tf.matmul(inputs, W) + b
    return a 

#because is 2D input:
# W = [[w1],[w2]] inputs = [x1, x2] or [x, y]
# prediction will be prediction = [[w1],[w2]] * [x, y] + b = w1*x + w2*y + b

#let's see loss function
#per_sample+losses is a tensor with the same shape as targets and prediction containing loss scores per sample
#we also need to average per-sample loss scores into a single scalar loss value, that is done by tf.reduce_mean 
# tf.reduce_mean(per_sample_losses) is the average of all values in the vector per_sample_losses
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions) 
    print('per sample losses =', per_sample_losses)
    q = tf.reduce_mean(per_sample_losses)   
    
    return q

#training step:
#first receives some training data and updates the weights W and b so to minimize the loss
learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)

    print('W =', W)
    print('b', b)
    print('predictions from model x*w+b = ', predictions)
    print('loss from reduce mean avg of (target-prediction)^2 = ', loss)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    print('Gradient loss of W ', grad_loss_wrt_W)
    print('Gradient loss of b ', grad_loss_wrt_b)
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    print('new W =', W)
    print('new b', b)

    return loss

#here we will not use mini-batch training to iterate the data into small batches. instead we will do batch training
#it will take longer to compute 2000 samples at once. But if we make it in small chunks it would be faster.
#But in the other hand the gradient update will be much more efficient at reducing the loss on the training data 
#if you choose to take fewer steps for training, you have to use larger learning_rate = 0.1, as in our example here.
#the batch training loop at 40 steps should look:
if __name__ == "__main__":
    for step in range(100):
        loss = training_step(inputs, targets)
        print(f"Loss at step {step}: {loss:.4f}")

#after 40 steps the training loss should be reduced.
#Because our targets are 0 and 1, each input point will be classified as "0" if the prediction is below 0.5 and "1" if it is above 0.5
    predictions = model(inputs)
    print('final prediction=', predictions)
    #plt.scatter(inputs[:,0], inputs[:,1], c=predictions[:,0] > 0.5)
    #plt.show()

    #so equation w1*x+w2*y+b < 0.5 will be classified 0
    #and w1*x+w2*y+b > 0.5 will be classified 1
    #now let make the equation of line w1*x+w2*y+b=0.5 to see the border of both
    #above the line will be class 1 and bellow the line class 0
    #Now let's' make some math: w1*x+w2*y+b=0.5 => w2*y=0.5 - b - w1*x => y= (0.5 - b - w1*x)/w2
    #"-r" plot it as red line
    x = np.linspace(-4,4,10)
    y = (5 - b - W[0]*x) / W[1]   
    plt.plot(x,y,"-r")   
    plt.scatter(inputs[:,0], inputs[:,1], c=predictions[:,0] > 5)
    plt.show()