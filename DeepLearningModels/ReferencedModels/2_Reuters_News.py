#this is an example of 46 mutually exclusive topics for Reuters news wires.
from tensorflow.keras.datasets import reuters

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

#For multiclass classification the loss function to use is “categorical crossentropy”. 
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
hist = model.fit(partial_x_train, 
                partial_y_train_labels, 
                epochs=20, 
                batch_size=512, 
                validation_data=(val_x, val_y)
                )

#now let's look the history dictionary containing data what will show us:
hist_details = hist.history

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

############################################################################
#After 9th epoch it begins to overfit
#now we need to train a new model from scratch for only 9 epochs and then we will evaluate to test set

model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
)

#Let's train the model for 9 epoch
model.fit(x_train_data, 
                y_train_labels, 
                epochs=9, 
                batch_size=512
                )

test_loss, test_accur = model.evaluate(x_test_data, y_test_labels)
print(f"test_acc: { test_accur } and loss: { test_loss }")

###########################################################################
#The accuracy will be approx: 80%. 
#let's see the accuracy for random baselin becouse here we have 46 classes
import copy
test_labels_copy = copy.copy(test_labels)
print('Before shuffling =', test_labels_copy)
np.random.shuffle(test_labels_copy)
print('After shuffling =', test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print('Final shuffling =', hits_array)
print('Mean =', hits_array.mean())

#random accuracy score would be around 17% classification accuracy, which mean that our model seems pretty good

##########################################################################
#Now let's generate topic prediction for all test data
probability = model.predict(x_test_data)
#each entry of predictions is of lenth 46
print('Shape of prediction= ', probability.shape)
#all coeficients of each sample the sum should be 1, because they form probability distribution
#let's see
print('Sum of all coeficients of just one random sample = ', np.sum(probability[33]))

#now let's see the largest entry of predicted class with the highest probability
print('Highest probability of one checked random 33th sample = ', np.argmax(probability[33]))

#predictions = np.zeros(np.shape(test_labels))
"""
for i in range(len(probability)):
    predictions[i] = np.argmax(probability[i])
"""
predictions = [np.argmax(probability[i]) for i in probability]
#predictions = [float(np.round(x)) for x in probabilities]

accuracy = np.mean(predictions == test_labels)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))