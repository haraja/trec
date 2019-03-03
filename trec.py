import numpy as np
from enums import Classification
import helpers
import mlfunc
import sys
import argparse
import time
import pickle
import matplotlib.pyplot as plt
from enums import Classification
import sys
import argparse

'''
TODO:
-fix relU functions - do not work currently
-implement regularization for cost function
-remove hardcodings from data read & array sizes
'''

# Read command line arguments. Possibility to select whether binary- or muticlass-clasification is used
#   binary classification: only learns single value from training data, and predicts only that single value
#   lears all numbers from training data and predicts any number value
parser = argparse.ArgumentParser(description='Harris machine learning script.')
parser.add_argument('-c', action='store', dest='classification_arg', default='binary', help='[binary | multiclass]')
args = parser.parse_args()

classification_type = Classification.BINARY
if args.classification_arg == 'multiclass':
    classification_type = Classification.MULTICLASS

# Get all training- and test-datasets
X, X_test, Y, Y_test = helpers.get_data(classification_type)
#helpers.show_number(X, mlfunc.convert_from_one_hot(Y), 59997)

weight_params = mlfunc.init_params(X, Y)
#print("weight params /1: ")
#print(weight_params)

# Either:
# 1. train model to get new weight parameters
start_time = time.time()
weight_params = mlfunc.run_model(X, Y, weight_params, 20, classification_type)
end_time = time.time()

# Or:
# 2. load weight parameters from earlier learned set
'''
with open('weight_params.pkl', 'rb') as f:
    weight_params = pickle.load(f)
'''

#print("weight params: /2 ")
#print(weight_params)

predictions = mlfunc.predict(X, weight_params, classification_type)
mlfunc.check_accuracy(Y, predictions)
print("time elapsed: " + str(end_time - start_time))
#print("predictions mean = " + str(np.mean(predictions)))

print("TEST SET")
predictions = mlfunc.predict(X_test, weight_params, classification_type)
mlfunc.check_accuracy(Y_test, predictions)

'''
# write learned weight parameters into disk
with open('weight_params.pkl', 'wb') as f:
    pickle.dump(weight_params, f)
'''