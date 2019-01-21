import numpy as np
from helpers import read_data, show_number
import mlfunc


'''
NEXT: 
-debug
-implement regularization for cost function
-remove hardcodings from data read & array sizes
'''


X = read_data("train-images.idx3-ubyte", 16)
Y = read_data("train-labels.idx1-ubyte", 8)
X_test = read_data("t10k-images.idx3-ubyte", 16)
Y_test = read_data("t10k-labels.idx1-ubyte", 8)

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

#show_number(X, Y, np.random.randint(0, Y.size))

# shape the output from labels to arrays
Y = mlfunc.labels_to_vectors(Y)
Y_test = mlfunc.labels_to_vectors(Y_test)

weight_params = mlfunc.init_params()

for i in range(500):
    cache_params = mlfunc.forward_propagation(X, weight_params)
    cost = mlfunc.compute_cost(Y, cache_params["A2"])
    #print("\nCost: " + str(cost))
    gradient_params = mlfunc.backward_propagation(X, Y, weight_params, cache_params)
    weight_params = mlfunc.update_params(weight_params, gradient_params)
    
    if i % 10 == 0:
        print("Cost: " + str(cost))

    mlfunc.check_accuracy(X, Y, weight_params)