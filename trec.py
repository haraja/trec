import numpy as np
from helpers import read_data, show_number
import mlfunc
import time
import pickle
import json
import matplotlib.pyplot as plt


'''
NEXT:
-debug
-implement regularization for cost function
-remove hardcodings from data read & array sizes
'''


X = read_data('train-images.idx3-ubyte', 16)
Y = read_data('train-labels.idx1-ubyte', 8)
X_test = read_data('t10k-images.idx3-ubyte', 16)
Y_test = read_data('t10k-labels.idx1-ubyte', 8)

# normalize the data 0..1
X = X/255
X_test = X_test/255

# shape the arrays for suitable dimensions
# images are 28x28 px -> in one dimension 1 image is 784 bytes long
X.shape = (60000, 784)
X_test.shape = (10000, 784)

# arrange numbers on columns - each column is separate number
X = X.T
X_test = X_test.T

# labels are arranged in row vectors
Y.shape = (1, Y.size)
Y_test.shape = (1, Y_test.size)

'''
temporarily change this for binary classification: Only search for number 5
'''
Y[Y != 5] = 0
Y[Y == 5] = 1
Y_test[Y_test != 5] = 0
Y_test[Y_test == 5] = 1


#show_number(X, Y, np.random.randint(0, Y.size))

# shape the output from labels to arrays
#Y = mlfunc.convert_to_one_hot(Y)
#Y_test = mlfunc.convert_to_one_hot(Y_test)

weight_params = mlfunc.init_params(X, Y)
#print("weight params /1: ")
#print(weight_params)

# Either:
# 1. train model to get new weight parameters
start_time = time.time()
weight_params = mlfunc.run_model(X, Y, weight_params)
end_time = time.time()

# Or:
# 2. load weight parameters from earlier learned set
'''
with open('weight_params.pkl', 'rb') as f:
    weight_params = pickle.load(f)
'''

#print("weight params: /2 ")
#print(weight_params)
#mlfunc.check_accuracy(X, Y, weight_params)
predictions = mlfunc.predict(X, weight_params)
mlfunc.check_accuracy(Y, predictions)
print("time elapsed: " + str(end_time - start_time))
#print("predictions mean = " + str(np.mean(predictions)))

print("TEST SET")
predictions = mlfunc.predict(X_test, weight_params)
mlfunc.check_accuracy(Y_test, predictions)

# write learned weight parameters into disk
with open('weight_params.pkl', 'wb') as f:
    pickle.dump(weight_params, f)
