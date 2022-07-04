#this is an example of 46 mutually exclusive topics for Reuters news wires.
from tensorflow.keras.datasets import reuters
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "Cats-vs-dogs-CNN-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#as in IMDB num_words=10000 restricts data to 10000 most frequently occuring words
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#8982 training examples and 2246 test examples
print('train samples shape: ', train_data.shape)
print('train labels shape: ', train_labels.shape)

print('test samples shape: ', test_data.shape)
print('test labels shape: ', test_labels.shape)

#look how are represented the list of integerst (word indexes)
print('train_data: ', train_data[10])

###################################################################################
#Decoding to words the same as imdb example
word_indx = reuters.get_word_index()
#reverse_word_index, it mean reverse it that mapping integer indices to words
reverse_word_index = dict([(val, key) for (key, val) in word_indx.items()])
#Decodes the review. Inices are offset by 3 becouse 0,1,2 are reserved indices for 0-"padding", 1-"start of sequence" and
#2-"unknown"
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_newswire)
##########################################################################
#from 0 to 45 topics there are defined in the label
print('Labels: ', train_labels[10])

#################################################################
#now lets vectorize the data
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

#############################################################
#let's vectorize labels data
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))

    for i, lab in enumerate(labels):
        results[i, lab] = 1.
    return results
#now let's vectorize labels:
y_train_labels = to_one_hot(train_labels)
y_test_labels = to_one_hot(test_labels)

print('y train labels: ', y_train_labels)

"""
#there is already the similar way by using built-in way in Keras
from tensorflow.keras.utils import to_categorical

y_train_labels = to_categorical(train_labels)
y_test_labels = to_categorical(test_labels)
"""
#############################################################
#now let's see how to use the model for this multi classification
from tensorflow import keras
from tensorflow.keras import layers
#in our previous example we got 16 dimensional intermediate layers, but here we have 46 different classes 
#and if we do the same thing each layer become an information bottleneck and will be too limited to learn 
#to separate 46 different classes and some information will be dropped
#so 64 dimensions will be the best choice for now for this example
#The final output will be 46 dimensional vector where each entry inside the vector will encode a different output class
model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax")
])

#It measures the distance between two probability distribution (between probability distribution output by the model 
#and the true distribution of the labels).
comp = model.compile(optimizer="rmsprop",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
)

############################################################
#Now let's set apart 1000 samples for validation
val_x = x_train_data[0:1000]
partial_x_train = x_train_data[1000:]
val_y = y_train_labels[0:1000]
partial_y_train_labels = y_train_labels[1000:]

############################################################
#Let's train the model
model.fit(partial_x_train, 
                partial_y_train_labels, 
                epochs=20, 
                batch_size=512, 
                validation_data=(val_x, val_y),
                callbacks=[tensorboard]
                )

#run this code in jypiter collaboratory
"""
%load_ext tensorboard
%tensorboard --logdir logs
"""