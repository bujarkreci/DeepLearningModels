# Sample Multilayer Perceptron Neural Network in Keras
#from keras.models import Sequential
#from keras.layers import Dense
import numpy as np
# load and prepare the dataset
filenm = open('PythonDeepLearning/Examples/Diabetes/pima-indians-diabetes.csv', 'rb')
dataset = np.loadtxt(filenm, delimiter=",")
X = dataset[:,0:8]
y = np.array(X)

size = len(y[0])
result = np.zeros((size,3))
print('result shape', result)
sd = []
for i in range(size):
    arr = y[:,i:i+1]
    meani = np.mean(arr)
    sd = np.std(arr)
    result[i,0]=i+1
    result[i,1]=round(meani,3)
    result[i,2]=round(sd,3)

print('result = ', result)

