'''
Machine learning functions
'''

import numpy as np
from enums import Classification


def sigmoid(Z):
    ''' Sigmoid activation function
    '''
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    ''' Derivative of Sigmoid activation function
    '''
    return Z * (1 - Z)


def tanh(Z):
    ''' Tanh activation function
    '''
    return np.tanh(Z)


def tanh_derivative(Z):
    ''' Derivative of Tanh activation function
    '''
    return 1 - np.power(Z, 2)


def relu(Z):
    ''' ReLU activation function
    '''
    return np.maximum(0, Z)


def relu_derivative(Z):
    ''' Dericative of ReLU activation function
    '''
    Z[Z >= 0] = 1
    Z[Z < 0] = 0
    return Z


def softmax(Z):
    '''Calculates softmax values for multiclass classification - output layer on neural network

    Args: Vector of Z-values of output layer

    Returns: Vector of A-values, calculated with softmax activation function 
    '''

    #n = Z.size
    #assert n == 10
    #A = np.zeros((n, 1))

    #divisor = 0
    divisor = np.sum(np.exp(Z))
    assert isinstance(divisor, float)
    #for i in range(n):
    #    divisor += np.exp(Z[i, 0])

    A = np.exp(Z) / divisor
    #for i in range(n):
    #    A[i, 0] = np.exp(Z[i, 0] / divisor
        #assert isinstance(A[i, 0], float)

    assert A.shape == (10, 1)
    return A
 

def convert_to_one_hot(Y_labels):
    '''Codes each label in array to character array to be used in neural network
    Each label will be represented by 10 character vector of 0's and 1
    This is known as 'one-hot' encoding
    In matrix this looks as follows:
        0 1 2 3 4 5 6 7 8 9
        ===================
        1 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 1

    Args:
    Y_labels -- array of labels

    Returns:
    Y_one_hot -- Array of all labels represented in one-hot format
    '''

    assert Y_labels.shape[0] == 1
    m = Y_labels.size
    Y_one_hot = np.zeros((10, m))

    for i in range(m):
        Y_one_hot[Y_labels[0, i], i] = 1

    return Y_one_hot


def convert_from_one_hot(Y_one_hot):
    '''Converts One Hot array back to labels
    '''

    n = Y_one_hot.shape[0]
    m = Y_one_hot.shape[1]
    assert n == 10

    Y_labels = np.zeros((1, m))

    for i in range(m):
        for j in range(n):
            if Y_one_hot[j, i] == 1:
                Y_labels[0, i] = j
                break

    return Y_labels


def init_params(X, Y):
    '''Initializes the parameter sof neural net

    Initially implemented for neural network with 1 hidden layer only
    (and one input and one output layer)

    Returns:
    weight_params -- dictionary containing all weight parameters
        W1 -- weight matrix of layer 1, shape: hidden_layer x input_layer
        b1 -- bias vector of layer 1, shape: hidden_layer x 1
        W2 -- weight matrix of layer 2, shape output_layer x hidden_layer
        b2 -- bias vector of layer 2, shape output_layer x 1
    '''

    # TODO: Get hardcoded values from below rather from image dimension etc.
    n_x = X.shape[0]    # size of input layer - size of 1 image
    n_h = 15            # size of hidden layer
    n_y = Y.shape[0]    # size of output layer
    # TODO: take following lines eventually out
    assert n_x == 784
    assert n_y == 1

    # initial weight parameters need to be random, in order for network to work
    # TODO check which multipliers to add for Wx randoms
    np.random.seed()
    W1 = np.random.rand(n_h, n_x) * 0.01    # weight multipliers for hidden layer
    W2 = np.random.rand(n_y, n_h) * 0.01    # weight multipliers for output layer
    # b can be 0 in beginning, there is no reason to randomize that
    b1 = np.zeros((n_h, 1))                 # bias multiplier for hidden layer
    b2 = np.zeros((n_y, 1))                 # bias multiplier for output layer

    weight_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return weight_params


def forward_propagation(X, weight_params):
    '''Forward propagation of the parameters

    Parameters:
    X -- matrix containing images
    weight_params -- dictionary containing weight parameters

    Returns:
    dictionary caching the result
    '''

    W1 = weight_params['W1']
    b1 = weight_params['b1']
    W2 = weight_params['W2']
    b2 = weight_params['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    m = X[1].size # number of samples
    assert Z1.shape == (len(W1), m)
    assert Z2.shape == (len(W2), m)
    assert A1.shape == (len(W1), m)
    assert A2.shape == (len(W2), m)

    cache_params = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return cache_params


def compute_cost(Y, A):
    '''Computes cost of for the forward propagation

    Args:
    Y -- true labels
    A -- output of the last layer's activation

    Returns:
    cost
    '''

    m = Y.shape[1]
    log_calc = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), (1 - Y))
    cost = -1/m * np.sum(log_calc)

    cost = np.squeeze(cost)
    assert(isinstance(cost, float))

    return cost


def compute_cost_softmax(Y, A):
    m = Y.shape
    log_calc = -np.sum(np.multiply(np.log(A), Y), axis=0)
    cost = 1/m * np.sum(log_calc)


def backward_propagation(X, Y, weight_params, cache_params):
    '''Backward propagation computes delta between true values and computed weighted values

    Args:
    X -- input parameters (images)
    Y -- true labels
    weight_params -- weight parameters
    cache_params -- Z, A, parameters computed during forward propagation

    Returns:
    gradient_params --  parameters of the gradients (weight - derivative)
    '''

    m = X.shape[1]

    # Get weights parameters from forward propagations
    W1 = weight_params['W1']
    W2 = weight_params['W2']

    # Get activation parameters
    A1 = cache_params['A1']
    A2 = cache_params['A2']

    # Calculate derivatives
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(A1)
    #dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(A1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    # NOTE: right now using sigmoid activation function is all layers.
    # If different activation functions would be used, then dZx would also need to change

    gradient_params = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    return gradient_params


def update_params(weight_params, gradient_params):
    '''Updates weight parameters from the gradient

    Args:
    X -- input array
    weight_params -- dictionary of weight and bias parameters

    Returns:
    weight_params -- updated weight and bias parameters
    '''

    learning_rate = 1.5

    # Get weights parameters
    W1 = weight_params['W1']
    b1 = weight_params['b1']
    W2 = weight_params['W2']
    b2 = weight_params['b2']

    # Get gradient parameters
    dW1 = gradient_params['dW1']
    db1 = gradient_params['db1']
    dW2 = gradient_params['dW2']
    db2 = gradient_params['db2']

    # Update weight parameters
    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate

    weight_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return weight_params

def run_model(X, Y, weight_params, iterations, classification_type):
    # weight_params = init_params(X, Y)

    print("Cost:")
    cost = 0
    
    for i in range(iterations):
        cache_params = forward_propagation(X, weight_params)

        if classification_type == Classification.BINARY:
            cost = compute_cost(Y, cache_params['A2'])
        #else:
        #    cost = compute_cost_softmax(Y, cache_params['A2'])
        if i == 0:
            cost_start = cost

        cost = compute_cost(Y, cache_params['A2'])
        gradient_params = backward_propagation(X, Y, weight_params, cache_params)
        weight_params = update_params(weight_params, gradient_params)

        if i % 100 == 0:
            print(cost)
    
    return weight_params


def predict(X, weight_params):
    cache = forward_propagation(X, weight_params)
    A2 = cache['A2']
    print(A2)

    # if value > 0.5, this is considered to be the number
    predictions = np.round(A2)
    print("predictions: ")
    print(predictions)
    #print("predictions mean: " + str(np.mean(predictions)))
    
    return predictions


def check_accuracy(Y, predictions):
    '''Checks the accuracy of trained network
    '''

    print('check_accuracy::')

    correct_ones = 0
    total_ones = 0
    for i in range(Y.shape[1]):
        if Y[0, i] == 1:
            total_ones += 1
            if predictions[0, i] == 1:
                correct_ones += 1

    print("total_ones: " + str(total_ones))
    print("correct_ones: " + str(correct_ones))