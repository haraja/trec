import numpy as np
import helpers
import mlfunc
import sys
import argparse
import time
import pickle
import matplotlib.pyplot as plt
import json


with open('config.json') as f:
    config = json.load(f)

# Read command line arguments
parser = argparse.ArgumentParser(description='Harris machine learning script.')
parser.add_argument('-s', action='store_true', dest='store_params', help='store learned params to file')
parser.add_argument('-r', action='store_true', dest='read_params', help='read learned parameters from file')
parser.add_argument('-i', type=str, dest='file_name', help='evaluate image [FILE_NAME] against learned parameter set')
#parser.add_argument('-i', action='store_true', dest='image_input', help='evaluate input_image.jpg against learned parameter set')
args = parser.parse_args()

# Get all training- and test-datasets, as numpy-arrays from mnist-data
X, X_test, Y, Y_test = helpers.mnist_to_array()
#helpers.show_number(X, mlfunc.convert_from_one_hot(Y), 2)

if args.read_params or args.file_name:
    # Read weight parameters from earlier learned set
    with open('weight_params.pkl', 'rb') as f:
        weight_params = pickle.load(f)
else:
    # Train model to get new weight parameters
    
    learning_rate = config['hyperparameters']['learning_rate']
    lambd = config['hyperparameters']['lambd']
    hidden_layer_dims = np.array(config['hyperparameters']['hidden_layer_dimensions'])
    iterations = config['other']['iterations']

    # n_h_dims only has hidden layer(s) dimension(s). Update this array to describe whole network:
    # input layer dimension in the beginning, output layer dimension in the end
    input_and_hidden_layers = np.insert(hidden_layer_dims, 0, X.shape[0])
    layer_dims = np.append(input_and_hidden_layers, Y.shape[0]) # input + hidden + output layer(s)
    
    start_time = time.time()
    weight_params = mlfunc.init_params(layer_dims)
    weight_params = mlfunc.run_model(X, Y, weight_params, iterations, learning_rate, lambd)
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
    predictions = mlfunc.predict(X, weight_params)
    mlfunc.check_accuracy(Y, predictions)
    #print('predictions mean = ' + str(np.mean(predictions)))
    print('TEST SET')
    predictions = mlfunc.predict(X_test, weight_params)
    mlfunc.check_accuracy(Y_test, predictions)

if args.store_params == True:
    # write learned weight parameters into disk
    with open('weight_params.pkl', 'wb') as f:
        pickle.dump(weight_params, f)