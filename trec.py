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


# Read command line arguments
#   binary classification: only learns single value from training data, and predicts only that single value
#   lears all numbers from training data and predicts any number value
parser = argparse.ArgumentParser(description='Harris machine learning script.')
parser.add_argument('-bc', action='store_true', dest='bin_classification', help='binary classification')
parser.add_argument('-s', action='store_true', dest='store_params', help='store learned params to file')
parser.add_argument('-r', action='store_true', dest='read_params', help='read learned parameters from file')
#TODO: Complete -i option form import image - compare this single image with learned parameters
parser.add_argument('-i', action='store_true', dest='image_input', help='evaluate input_image.jpg against learned parameter set')
args = parser.parse_args()

classification_type = Classification.MULTICLASS
if args.bin_classification == True:
    classification_type = Classification.BINARY

# Get all training- and test-datasets, as numpy-arrays from mnist-data
X, X_test, Y, Y_test = helpers.mnist_to_array(classification_type)
#helpers.show_number(X, mlfunc.convert_from_one_hot(Y), 2)

weight_params = mlfunc.init_params(X, Y)

if args.read_params or args.image_input:
    # Read weight parameters from earlier learned set
    with open('weight_params.pkl', 'rb') as f:
        weight_params = pickle.load(f)
else:
    # Train model to get new weight parameters
    start_time = time.time()
    weight_params = mlfunc.run_model(X, Y, weight_params, 2000, classification_type)
    end_time = time.time()
    print('time elapsed: ' + str(end_time - start_time))

if args.image_input:
    X_from_input = helpers.jpg_to_array()
    X_reshaped = X_from_input
    X_reshaped.shape = (28, 28)
    plt.imshow(X_reshaped, cmap='gray')
    plt.show()
    
    print('INPUT FILE')
    prediction, propability = mlfunc.predict_single(X_from_input, weight_params)
    print('Number is: %d, propability: %f' % (prediction, propability*100))
else:
    print('TRAINING SET')
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