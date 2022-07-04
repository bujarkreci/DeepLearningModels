# Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# load and prepare the dataset
filenm = open('PythonDeepLearning/Examples/Diabetes/pima-indians-diabetes.csv', 'rb')
dataset = np.loadtxt(filenm, delimiter=",")
X = dataset[:,0:8]
print('X shape ',X.shape)
print('X = ',X)
Y = dataset[:,8]
print('Y shape', Y.shape)
print('Y= ', Y)
# 1. define the network
model = Sequential()
model.add(Dense(16, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 2. compile the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3. fit the network
history = model.fit(X, Y, epochs=150, batch_size=16)
# 4. evaluate the network
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5. make predictions
probabilities = model.predict(X)
print('Probability=', probabilities)
predictions = [float(np.round(x)) for x in probabilities]
print('predictions=', predictions)
accuracy = np.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))