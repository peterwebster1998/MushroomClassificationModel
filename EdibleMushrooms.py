# A final project for the Neural Networks Directed study:
# Instructor: Professor Haluk Ogmen
# Student/Author: Peter Webster
# Winter Quarter 2020 - University of Denver RSECS

import csv
from keras.models import Sequential
import keras.layers as layers
from keras.losses import CategoricalCrossentropy
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

K.common.set_image_dim_ordering('tf')

# Fetch data from .csv file
dataset = open('mushrooms.csv')
dataread = csv.reader(dataset, delimiter=',')

# Transform data into a readable format
rows, cols = (8125, 23)
data = [['null' for i in range(cols)] for j in range(rows)]
i, j = (0, 0)
for line in dataread:
    for datum in line:
        data[j][i] = datum
        i += 1
    j += 1
    i = 0

# Transform data into categorical data
#numCateg = [2, 6, 4, 10, 2, 9, 4, 3, 2, 12, 2, 7, 4, 4, 9, 9, 2, 4, 3, 8, 9, 6, 7]  # some categories not present in dataset
# Create dictionary of all possible variants of all features
categories = {}
for i in range(len(data[0])):
    iter = 0
    for datum in data[1:]:
        if iter != 0:
            if not categories[data[0][i]].__contains__(datum[i]):
                categories[data[0][i]].append(datum[i])
        else:
            categories[data[0][i]] = [datum[i]]
        iter += 1

# Use feature dictionary to convert categorical data into 1s & 0s
# Count features
numFeatures = 0
for key in categories:
    numFeatures += len(categories[key])
print('Num Features: ', numFeatures)

# Convert datatype
newData = np.zeros([len(data) - 1, numFeatures])
# Iterate through lines of dataset
k = 0
for line in data[1:]:
    i, j = (0, 0)
    # Iterate through categories
    for cat in categories:
        # Iterate through features of each category
        for entry in categories[cat]:
            newData[k][j] = 1*(line[i] == entry)
            j += 1
        i += 1
    k += 1
data = newData

# Split dataset into Training, Validation & testing data with labels
Train = np.zeros([5000, numFeatures - 2])
Train_Edibility = np.zeros([5000, 2])
Test = np.zeros([1000, numFeatures - 2])
Test_Edibility = np.zeros([1000, 2])
Validation = np.zeros([data.shape[0] - 6000, numFeatures - 2])
Validation_Edibility = np.zeros([data.shape[0] - 6000, 2])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if i < 1000:
            # Test data
            if j < 2:
                Test_Edibility[i][j] = data[i][j]
            else:
                Test[i][j-2] = data[i][j]

        elif i < 6000:
            # Training data
            if j < 2:
                Train_Edibility[i-1000][j] = data[i][j]
            else:
                Train[i-1000][j-2] = data[i][j]
        else:
            # Validation data
            if j < 2:
                Validation_Edibility[i-6000][j] = data[i][j]
            else:
                Validation[i-6000][j-2] = data[i][j]

'''
Bizarre indexing issues, issue resolved using nested loop above but slows instantiation of data
Train = data[:5002][2:]    # Excludes feature labels
Validation = data[5001:6003][2:]    # Smaller dataset for validation
Test = data[6002:][2:]      # Final values used for test dataset

# Store the edibility of each sample for training validation & testing
Train_Edibility = data[:5001][:2]
Validation_Edibility = data[5001:6002][:2]
Test_Edibility = data[6002:][:2]
'''


def plot_exp1(losses, x):
    # fig, ax = plt.subplot()
    plt.plot(x, losses[0], '-ro', label='Tanh', linewidth=2.0)
    plt.plot(x, losses[1], '-bo', label='Sigmoid', linewidth=2.0)
    plt.plot(x, losses[2], '-go', label='ReLU', linewidth=2.0)

    plt.axis([1, 10, 0, 1.0])
    plt.title('256 Neurons per Layer w/ Dropout')
    plt.xlabel('Hidden Layers')
    plt.ylabel('Cross Entropy Loss')
    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()

    plt.show()

# define the larger model
def larger_model(hidden_layers, nodesPerLayer, actFun):
    # create model
    model = Sequential()
    # '''complete the structure'''
    for i in range(0, hidden_layers):
        if i == 0:
            model.add(layers.Dense(nodesPerLayer[i], input_dim=(numFeatures-2), activation=actFun))
        else:
            # Fully connected layer
            model.add(layers.Dropout(0.1))
            model.add(layers.Dense(nodesPerLayer[i], activation=actFun))
    # Softmax exit layer
    model.add(layers.Dense(2, activation='softmax'))
    # Compile model. Use cross entropy as loss function and the Adam optimizer!
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model


#acts = ['tanh', 'sigmoid', 'relu']
#hidden_layers = [1, 2, 3, 4, 5, 6, 8, 10]
#losses = np.zeros([3, 8])
#i = 0
#for act in acts:
#    j = 0
#    for layer in hidden_layers:
#        print('\nActFun: ', act, ' Hidden Layers: ', layer, '\n')
# Create network
model = larger_model(6, [128, 128, 32, 32, 8, 8], 'tanh')
# Fit the model
model.fit(Train, Train_Edibility, validation_data=(Validation, Validation_Edibility), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(Test, Test_Edibility, verbose=0)
print("Network Error: %.2f%%" % (100 - scores[1] * 100))
#        losses[i][j] = scores[0]
#        j += 1
#    i += 1
#plot_exp1(losses, hidden_layers)
