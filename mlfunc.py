'''
Machine learning functions
'''


import numpy as np


def sigmoid(Z):
    '''Sigmoid activation function. Works with numbers and matrices

    Args:
    Z -- real value / matrix

    Returns:
    float number / matrix, activation with sigmoid function
    '''

    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    return Z * (1 - Z)


def tanh(Z):
    return np.tanh(Z)


def tanh_derivative(Z):
    return 1 - np.power(Z, 2)


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    Z[Z >= 0] = 1
    Z[Z < 0] = 0
    return Z


def convert_to_one_hot(Y_labels):
    '''Codes each label in array to character array to be used in neural network
    Each label will be represented by 10 character vector of 0's and 1
    This is known as One Hot encoding
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
    Y_vectors -- Array of all labels represented in One Hot format
    '''

    m = Y_labels.size
    Y_vectors = np.zeros((10, m))

    for i in range(m):
        Y_vectors[Y_labels[0, i], i] = 1

    return Y_vectors


def convert_from_one_hot(Y_vectors):
    '''Converts One Hot array back to labels
    '''

    n = Y_vectors.shape[0]
    m = Y_vectors.shape[1]

    Y_labels = np.zeros((1, m))

    for i in range(m):
        for j in range(n):
            if Y_vectors[j, i] == 1:
                Y_labels[0, i] = j
                break

    return Y_labels



def init_params():
    '''Initializes the parametersof neural net

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
    n_x = 784   # size of input layer - size of 1 image
    n_h = 15    # size of hidden layer
    n_y = 10    # size of output layer, number of labels (possible characters)

    #NOTE: it's not nocessary to initialize b values randomly. 0 is ok
    np.random.seed()
    W1 = np.random.rand(n_h, n_x)   # weight multipliers for hidden layer
    W2 = np.random.rand(n_y, n_h)   # weight multipliers for output layer
    b1 = np.zeros((n_h, 1))         # bias multiplier for hidden layer
    b2 = np.zeros((n_y, 1))         # bias multiplier for output layer
    #b1 = np.random.rand(n_h, 1)
    #b2 = np.random.rand(n_y, 1)

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
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

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

    m = Y.shape[1]  # Number or y-units (0..9)

    log_calc = np.multiply(np.log(A), Y) + np.multiply(np.log(1-A), (1-Y))
    cost = -1/m * np.sum(log_calc)
    # print('compute_cost::cost ' + str(cost))

    return cost


def backward_propagation(X, Y, weight_params, cache_params):
    '''Backward propagation computes delta between true values and computed weighted values

    Args:
    X -- input parameters (images)
    Y -- true labels
    weight_params --
    cache_params -- Z, A, parameters computed during forward propagation

    Returns:
    gradient_params --  parameters of the gradient (weight - derivative)
    '''

    m = Y.shape[1]  # Number or y-units (0..9)

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
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(A1)
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

    learning_rate = 0.05

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


def check_accuracy(X, Y, weight_params):
    '''Checks the accuracy of trained network
    '''

    m = X.shape[1]

    Y_predictions = np.zeros((1, m))

    cache_params = forward_propagation(X, weight_params)
    A2 = cache_params['A2']
    A2_labels = convert_from_one_hot(A2)

    predict_vector = np.argmax(A2, axis = 0)
    predict_vector.shape = (1, m)

    print('check_accuracy::')
    print('A2:')
    print(A2.shape)
    for i in range(10):
        print(A2[i, 0])
        print(A2[i, 1])
    print('predict_vector:')
    print(predict_vector.shape)
    print(predict_vector)
