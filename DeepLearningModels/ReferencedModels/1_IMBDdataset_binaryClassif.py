#we get a sample of 50000 highly polarized reviews from IMDB.
#they are split into 25000 reviews for testing and 25000 others for training.
#each set consist 50% positive and 50% negative reviews.
#IMDB dataset comes packed with Keras, the same as MNIST handwritten digits.
#all the reviews has been preprocessed (ex: som sequence of words have been turned to integers where each integer 
##stand for specific word in a dictionary) In chapter 11 you will learn how to process raw text input.
from tensorflow.keras.datasets import imdb
#argument num_words=10000 means we keep only top 10000 most frequently occuring words in training data. Other rare words will be 
#eliminated. If we are not specifying num_words=10000 we will have stuck with more than 88585 unique words in training data,
#which is very large and unnecessary to get all of them to process. Some of those unique words are used only in a single sample,
#which is not important for us.
#train_data and test_data are lists of reviews (word indexec or uid - encoding a sequence of a words).
#train_labels and test_labels are list of 0s and 1s, where 0 mean false, negative, no etc and 1 mean positive, yes, excellent etc
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#print some uid of words and check their answers 1-positive 0-negative
print('Words reviews uid = ', train_data[0])
print('0-negative, 1-positive answer: ', train_labels[0])
print('train samples: ', train_data[0:5])
print('train labels: ', train_labels[0:5])

print(train_data.shape,'/n')
print(train_labels.shape)

#let's verify that there are is not higher than 9999 index number word. With this we conclude that there are restricted 10000 most frequent words
print(max([max(seq) for seq in train_data]))

#####################################################
#Let's decode one of the reviews back to English words:
#word_index is dictionary mapping words to an integer index.
word_index = imdb.get_word_index()
#reverse_word_index, it mean reverse it that mapping integer indices to words
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
#Decodes the review. Inices are offset by 3 becouse 0,1,2 are reserved indices for 0-"padding", 1-"start of sequence" and
#2-"unknown"
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_review)

####################################################################
#preparing data
#Lists of integers into a neural networks have to be in the same length.
#imdb data currently does not have the same shape after encoding.
#one way to do that is by padding(mbushi) your lists, to be in the same lenth (samples, max_length)
#second way is by multi-hot encode to make it the list to turn them into vectors of 1s and 0s.
#example sequence [8,5] will be turned in 10000 dimensional vector with all 0s except indices 8 and 5 which will be 1s.
# and after this it could be used Dense layer to handle floating point vector data
#let's se it:
#multi-hot encoding
#some of lists have different length, so you can check some:
for j in range(5):
    print('List,',j,' length: ', len(train_data[j]))

import numpy as np

def vectorize_seq(sequences, dimension=10000):
    #it creates the matrix with 0s in shape 25000x10000 currently for our problem
    results = np.zeros((len(sequences), dimension))
    #enumerate is ex: list = ["eat", "sleep", "repeat"] it creates [(0, "eat"), (1, "sleep"), (2, "repeat")]
    #sets 1s to the indexes that have specified the indexes of words i-"is the review number" and j-"is the word index"
    for i, sequ in enumerate(sequences):
        for j in sequ:
            results[i, j]=1
    return results 
#need to be vectorized both, the trained data and test data
x_train_data = vectorize_seq(train_data)
x_test_data = vectorize_seq(test_data)

#print and check the example
print('mapped to 0s and 1s=', x_train_data[0])

#now let's vectorize labels:
y_train_labels = np.asarray(train_labels).astype("float32")
y_test_labels = np.asarray(test_labels).astype("float32")

print('y train labels: ', y_train_labels)

#now the data is ready to be fed in neural network.
#here the input data are vectors and labels are scalars(0s and 1s).

###################################################################################
#now let's see how to use the model for this binary classification
#So the choice of the layers should look like Dense(units=16)->Dense(units=16)->Dense(units=1)
from tensorflow import keras
from tensorflow.keras import layers

#remember that each Dense layer with relu activation implements this: output = relu(dot(input,W)+b)
#so the W will have the dimension (inputDim, 16), where the dot product will project the dimension onto 16-dimensional representation
#the final layer layers.Dense(1, activation="sigmoid") have the main goal to outputs the probability 
#(1 or 0 indicating "how likely the review is to be pos or neg")
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
    ])

#after you define layers, it is time to compile the optimizer, loss and choose metrics for our model
comp = model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )

#######################################################################################
#after you define layers and compiled your model, it is time before training of your data to validate first some samples
#you don't have to choose all samples, but choose some of them. Here example 10000 samples.
x_validating = x_train_data[0:10000]
partial_x_train = x_train_data[10000:]
y_validating = y_train_labels[0:10000]
partial_y_train = y_train_labels[10000:]

#let's now train the model for 20 epochs (20 iterationsover all samples in the training data) packed in 512 samples.
#in the same moment we monitor loss and accuracy on 10000 samples that we already set
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_validating, y_validating)
    )

#now let's look the history dictionary containing data what will show us:
hist_details = history.history

#keys() contains 4 entries: "accuracy", "loss", "val_accuracy" and "val_loss"
print('History keys: ', hist_details.keys())
print('history details:', hist_details)
#######################################################################################
#let's see the loss with the help of matplotlib
#"bo" is blue dot and "b" is blue line and "r" is red line
import matplotlib.pyplot as plt

loss_values = hist_details["loss"]
validated_loss_values = hist_details["val_loss"]
epochs = range(1,len(loss_values)+1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, validated_loss_values, "r", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#now the same for accuracy
plt.clf()
accur = hist_details["accuracy"]
val_acc = hist_details["val_accuracy"]
plt.plot(epochs, accur, "ro", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

###################################################################################
#on the 4th epoch we had overfitting becouse of validation data
#let's fix from the 4th epoch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
    ])

comp = model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )

history = model.fit(
    x_train_data,
    y_train_labels,
    epochs=4,
    batch_size=512
    )

test_loss, test_accur = model.evaluate( x_test_data, y_test_labels)
print(f"test_acc: { test_accur } and loss: { test_loss }")
#but again this approach achieves an accuracy of 88% and we need to be close to 95% at least.

####################################################################################
#now let's show our predicts
pred = model.predict(x_test_data)
print('prediction for all= ', pred)
print('prediction for [first index]= ', pred[0])
print('prediction argmax =  ',pred[0].argmax())
print('check if test_labels[0]=predictions[0].argmax() => ',y_test_labels[0])

