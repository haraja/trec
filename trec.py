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
import json


with open('config.json') as f:
    config = json.load(f)

# Read command line arguments
#   binary classification: only learns single value from training data, and predicts only that single value
#   multiclass classification lears all numbers from training data and predicts any number value
parser = argparse.ArgumentParser(description='Harris machine learning script.')
parser.add_argument('-bc', action='store_true', dest='bin_classification', help='binary classification')
parser.add_argument('-s', action='store_true', dest='store_params', help='store learned params to file')
parser.add_argument('-r', action='store_true', dest='read_params', help='read learned parameters from file')
parser.add_argument('-i', type=str, dest='file_name', help='evaluate image [FILE_NAME] against learned parameter set')
#parser.add_argument('-i', action='store_true', dest='image_input', help='evaluate input_image.jpg against learned parameter set')
args = parser.parse_args()

classification_type = Classification.MULTICLASS
if args.bin_classification == True:
    classification_type = Classification.BINARY

# Get all training- and test-datasets, as numpy-arrays from mnist-data
X, X_test, Y, Y_test = helpers.mnist_to_array(classification_type)
#helpers.show_number(X, mlfunc.convert_from_one_hot(Y), 2)

if args.read_params or args.file_name:
    # Read weight parameters from earlier learned set
    with open('weight_params.pkl', 'rb') as f:
        weight_params = pickle.load(f)
else:
    # Train model to get new weight parameters
    
    learning_rate = config['hyperparameters']['learning_rate']
    lambd = config['hyperparameters']['lambd']
    network_layers = config['hyperparameters']['network_layers']
    n_hidden_layer = config['hyperparameters']['n_hidden_layer']
    iterations = config['other']['iterations']
    
    start_time = time.time()
    weight_params = mlfunc.init_params(X.shape[0], Y.shape[0], n_hidden_layer, network_layers)
    weight_params = mlfunc.run_model(X, Y, weight_params, iterations, learning_rate, lambd, classification_type)
    end_time = time.time()
    print('time elapsed: ' + str(end_time - start_time))

if args.file_name:
    X_from_input = helpers.jpg_to_array(args.file_name)
    print(X_from_input.shape)
    X_reshaped = np.copy(X_from_input)
    X_reshaped.shape = (28, 28)
    plt.imshow(X_reshaped, cmap='gray')
    plt.show()
    
    print('INPUT FILE')
    print(X_from_input.shape)
    prediction, propability = mlfunc.predict_single(X_from_input, weight_params)
    print('Number is: %d, propability: %f' % (prediction, propability*100))
    #print(X_from_input)
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