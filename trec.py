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
import argparse


# Read command line arguments. Possibility to select whether binary- or muticlass-clasification is used
#   binary classification: only learns single value from training data, and predicts only that single value
#   lears all numbers from training data and predicts any number value
parser = argparse.ArgumentParser(description='Harris machine learning script.')
parser.add_argument('-bc', action='store_true', dest='bin_classification', help='binary classification')
parser.add_argument('-s', action='store_true', dest='store_params', help='store learned params to file')
parser.add_argument('-r', action='store_true', dest='read_params', help='read learned parameters from file')
args = parser.parse_args()

classification_type = Classification.MULTICLASS
if args.bin_classification == True:
    classification_type = Classification.BINARY

# Get all training- and test-datasets
X, X_test, Y, Y_test = helpers.get_data(classification_type)
#helpers.show_number(X, mlfunc.convert_from_one_hot(Y), 59997)

weight_params = mlfunc.init_params(X, Y)

if args.read_params == False:
    # Train model to get new weight parameters
    start_time = time.time()
    weight_params = mlfunc.run_model(X, Y, weight_params, 2000, classification_type)
    end_time = time.time()
    print('time elapsed: ' + str(end_time - start_time))
else:
    # Load weight parameters from earlier learned set
    with open('weight_params.pkl', 'rb') as f:
        weight_params = pickle.load(f)

predictions = mlfunc.predict(X, weight_params, classification_type)
mlfunc.check_accuracy(Y, predictions)
#print('predictions mean = ' + str(np.mean(predictions)))

print('TEST SET')
predictions = mlfunc.predict(X_test, weight_params, classification_type)
mlfunc.check_accuracy(Y_test, predictions)

if args.store_params == True:
    # write learned weight parameters into disk
    with open('weight_params.pkl', 'wb') as f:
        pickle.dump(weight_params, f)
