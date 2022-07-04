
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.datasets import mnist

#######################################################################
#This part is the input data
#input images are stored in Numpy tensors formated as float32 at shape(60000, 784) training data and (10000, 784) test data
(training_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(training_images.shape)
print(len(train_labels))
print(train_labels)

# here is test data
print(test_images)
print(len(test_labels))
print(test_labels)

training_images = training_images.reshape((60000, 28*28))
training_images = training_images.astype("float32") / 255 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

#until here
#######################################################################
#now this is our model
#this model consists of a chain of 2 dense layers which applies simple tensor operations to the input data
#these operations involve weight tensors which are in fact attributes of layers where the knowlegde of the model exist.
#output = activation(dot(W, input) + b) -> (RELU => (y = max(0,x) -> df/dx = 1 if x>=0 else 0 ) for the first and SOFTMAX for the second layer)
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
    ])

#until here
#######################################################################
#this is the model compilation step
#sparse_categorical_crossentropy is the loss function used as a feedback signal for learning the weight tensors. And which the trainig
#phase will attempt to minimize
#reduction of the loss happens via mini-batch random gradinet descent.
#rules of gradient descent are defined by rmsprop optimizer

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#until here
#######################################################################
#this is the training loop
#with fit operation, The model will start to iterate on the training data in minibatches of 128 samples, 
##5 times (each iteration over all training data)
#on each minibatch the model will compute the gradient of the loss with regard to the weights (by backpropagation algorithm)
#and then it will move the weights in the direction that will reduce the value of loss for one batch.
#after 5 epochs, the model performed 2345 gradient updates in total (469 per epoch)
history = model.fit(training_images, train_labels, epochs=5, batch_size=128)

######################################################################

#predict probabilities now
test_digits = test_images[0:10]
print('test digits= ',test_digits)
predictions = model.predict(test_digits)
print('prediction for [first index]= ', predictions[0])
#when checking the highest probability score, that must be the number that we want  = index with the highest score
# at our example, the highest score is at index 7 with probability score 9.9994242e-1 almost 1, so the number we looking for is 7
#now check
print('prediction argmax =  ',predictions[0].argmax())
#now check the predictions
print('predictions[0][7] = ',predictions[0][7])

#now check if the test labels agrees if test_labels[0]=predictions[0].argmax()
print('check if test_labels[0]=predictions[0].argmax() => ',test_labels[0])

#now check how good is classifying digits?
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: { test_acc } and loss: { test_loss }")

#if you want to check the history metric data such as "loss", "accuracy" or others
print(history.history)

#as you see test_acc (0.978) < (a bit lower than) training set (0.9887) and this is becouse of overfitting