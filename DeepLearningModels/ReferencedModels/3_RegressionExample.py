#Boston Housing price
#attempt to predict the median price of homes in Boston suburb
#data points consist some features (13), some are: per capita crime rate, avg numbers of rooms, accessibility to highways, time etc.
#dataset consist 404 training samples and 102 test samples
from tensorflow.keras.datasets import boston_housing
import numpy as np

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print('Train data shape =', train_data.shape)
print('Test data shape = ', test_data.shape)
print('train targets shape =', train_targets.shape)

#training data consist 13 features described above (per capita crime rate, avg numbers of rooms, accessibility to highways, time etc.)
print('Train data =', train_data)

#these targets are median values of owner-occupied homes, in thausands of dollars typically between $10,000 and $50,000
#this data are in mid 1970 and this prices are not adjusted for inflation
print('Train targets = ', train_targets)

###############################################################################
#for each of 13 features we find z score z = (X-mean)/(standard deviation)
#axis=0 defines the mean for each column (axis=1 defines for each row)
meani = train_data.mean(axis=0)
print('mean for each column: ',meani)
train_data -= meani

#the same will do to find standard deviation for each column
stdi = train_data.std(axis=0)
print('standard deviation for each column: ', stdi)
train_data /= stdi

#beware: never compute quantity like mean or std in test_data, you should do it in training_data only
#because your are doing prediction based on training data and not test data
test_data -= meani
test_data /= stdi
print('Train data after preprocessing by finding z score=', train_data)

#these targets are median values of owner-occupied homes, in thausands of dollars typically between $10,000 and $50,000
#this data are in mid 1970 and this prices are not adjusted for inflation
print('Test data after preprocessing by finding z score= ', test_data)
##############################################################################
#because there aro only 506 samples in total, it is enough to use only 2 intermediate layers each with 64 units
#if you use more layers you will have worse overfitting, try to use small model
from tensorflow import keras
from tensorflow.keras import layers

#when you need to instantiate the same model multiple times, it is better to define it in a function to construct it
#model with single unit without activation, that means that this is a linear layer or scalar regression becouse you try to predict
#a single continous value
#with sigmoid you predict value betwee 0 and 1, but here this is simple linear and model is free to learn to predict
#values in any range
#loss - mse - (mean squared error), it means the square of difference between predictions and targets
#this is recommended for regression problems
#metric used: mae - (mean absolute error): this is the absoulute value of the difference between predictions and targets
#ex: mae = 0.5 mean prediction are off by $500 on average.
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1) #there is no need an activation function becouse it is scalar regression (need to predict range of amount of money)
        ])

    model.compile(
        optimizer="rmsprop",
        loss="mse", #squared difference between predicted value and target (predicted - target)^2
        metrics=["mae"] #absolute value of difference between prediction and target value. Ex: mae=0.5 is $500 off in predicting in average
        )
    return model

##############################################################################
#for very small samples approx hundreds, it is not good to split data for training set and validating set as we did in IMDB
#and in reuters example, becouse validation score will change a lot depending on which data point we choose for validation
#and which we choose for training, it might have high variance with regard to validation split.
#the best practice is to use K-fold cross validation method for such a small samples
#train_data=406/100 = 4, that's why k=4
k=4
num_val_samples = len(train_data) // k 
num_epochs = 100
all_scores = []
#verbose=0 mean training model in silent mode
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i*num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i*num_val_samples: (i+1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print('Let''s see the scores: ', all_scores)
print('The mean of all scores: ', np.mean(all_scores))
#mean will show approx: 2.2913. So it means we are of by $2300 on average. Significant for prices between $10000 and $50000

#################################################################################
#let''s try for 500 epochs
k=4
num_val_samples = len(train_data) // k 
num_epochs = 500
all_mae_historyscores = []
#verbose=0 mean training model in silent mode
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i*num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i*num_val_samples: (i+1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=16, verbose=0)
    mae_hist = history.history["val_mae"]
    all_mae_historyscores.append(mae_hist)

xi = np.array(all_mae_historyscores)
print('Shape of xi = ', xi.shape)
avg_mae_hist = [np.mean([x[i] for x in xi]) for i in range(num_epochs)]
print('Average mae history: ', xi)
yi = np.array(avg_mae_hist)
print('Avg MAE History shape = ', yi.shape)
print('yi = ', yi)

#####################################################################################
#let us plot this:
import matplotlib.pyplot as plt
epochsi = range(1,len(yi)+1)
plt.plot(epochsi, yi, "b", label="validation MAE by epoch")
plt.title("Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.legend()
plt.show()
#################################################################################
#first 10 data points are in different scale and let us omit those
omit_10_highranged = yi[10:]
epochsi = range(1,len(omit_10_highranged)+1)
plt.plot(epochsi, omit_10_highranged, "b", label="validation MAE by epoch")
plt.title("Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.legend()
plt.show()

###################################################################################
#it shows that validation MAE stops improving after 170-190 epochs which includes our 10 omitted epochs
# and this is our final best improvement
model = build_model()
#170 + 10 ommitted = 180 epochs
model.fit(train_data, train_targets, epochs=180, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(f"test MSE score: { test_mse_score } and test MAE score: { test_mae_score }")
#mean will show approx: 2.316. So it means we are of by $2300 again on average. Significant for prices between $10000 and $50000

###################################################################################
#now if we predict we will retrieve scalar score between 0 and 1
#model's guess for the sample's price in thousand of dollars
predictions = model.predict(test_data)
print('sample 0, let''s see prediction ', predictions[0])
#predictions[0] = 9.044375
#first house of our test set is predicted to have a price of $9044.375