'''
Machine learning functions
'''

import numpy as np
from enums import Classification
import matplotlib.pyplot as plt


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
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


def softmax(Z):
    '''Calculates softmax values for multiclass classification - output layer on neural network
    Z -> softmax-actication -> A

    Args: Vector of Z-values of output layer

    Returns: Vector of A-values, calculated with softmax activation function
    '''
    assert Z.shape[0] == 10

    m = Z.shape[1]
    divisor = np.sum(np.exp(Z), axis=0)
    divisor.shape = (1, m)

    tZ = np.exp(Z)
    A = np.divide(tZ, divisor)

    #TODO: check whether this could be done just with 1 line of code: 
    # np.exp(Z) / np.sum(np.exp(Z), axis=0)

    return A


def convert_to_one_hot(Y_labels):
    '''Codes each label in array to character array to be used in neural network
    As this program works with numbers 0..9, each label will be represented by 10 character vector of 0 and 1
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
    Y_labels -- 1 x m array of labels. Each label is single number

    Returns:
    Y_one_hot -- array of labels - each label represented in one-hot format
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
    Y_labels = np.zeros((1, m))
    
    assert n == 10

    for i in range(m):
        for j in range(n):
            if Y_one_hot[j, i] == 1:
                Y_labels[0, i] = j
                break

    return Y_labels


def init_params(layer_dims):
    '''Initializes the parameter for neural net

    Parameters:
        layer_dims -- array of neural-network layer dimensions. Notice that this includes all layers.
                      Size of input layer is size (dimensions) of one sample (one image)
                      Size of hidden layers depends of configuration. Can have 1..n hidden layers
                      Size of output layer is number of label categories
                      Example: [378, 20, 20, 10]

    Returns: weight_params -- dictionary containing all weight parameters
        W1 -- weight matrix of layer 1, shape: hidden_layer x input_layer
        b1 -- bias vector of layer 1, shape: hidden_layer x 1
        ...
        Wn
        bn
    '''

    '''
    print('layer_dims: ')
    print(len(layer_dims))
    for i in range(len(layer_dims)):
        print(layer_dims[i])
    exit(0)
    '''
    
    assert layer_dims[0] == 784                     # input layer - only works with images which have 784 (28x28) pixels
    assert layer_dims[len(layer_dims) - 1] == 10    # output layer
    #assert (layer_dims[2] == 1) or (layer_dims[2] == 10)    # really bad way to test. This works with binary- and multiclass-classfication
    
    # TODO: why is next 2 lines working?
    #if __debug__:
    #    np.random.seed(1)
    np.random.seed(1)

    weight_params = {}

    for layer in range(1, layer_dims.size):
        l_prev = layer_dims[layer - 1]
        l_current = layer_dims[layer]
        # np.sqrt... multiplier in the end is 'Xavier initialization'. 
        # This set the scale of params so that it helps to avoid vanishing/exploding gradients.
        # With relu, this should be np.sqrt(2/n_x)
        weight_params['W' + str(layer)] = np.random.rand(l_current, l_prev) * np.sqrt(1/(l_prev))
        weight_params['b' + str(layer)] = np.zeros((l_current, 1))

        assert(weight_params['W' + str(layer)].shape == (l_current, l_prev))
        assert(weight_params['b' + str(layer)].shape == (l_current, 1))


    assert(len(weight_params) == (layer_dims.size - 1) * 2)

#    print('weight params:')
#    print(weight_params)
#    exit(0)
    return weight_params


def forward_propagation(X, weight_params, classification_type=Classification.MULTICLASS):
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

    if classification_type == Classification.BINARY:
        A2 = sigmoid(Z2)
    else:
        A2 = softmax(Z2)    # for multiclass classification

    m = X[1].size # number of samples
    assert Z1.shape == (len(W1), m)
    assert Z2.shape == (len(W2), m)
    assert A1.shape == (len(W1), m)
    assert A2.shape == (len(W2), m)

    cache_params = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return cache_params


def forward_propagation_deep(X, weight_params):
    '''
    '''    
    cache_params = {}
    A_prev = X
    L = len(weight_params) // 2 # number of layer in neural net (excluding input-layer)
    assert(L == 2)

    # Forward propogation for all hidden layers - not for output layer
    # TODO: difference is only on calculating 'A', so these could be combined better
    for l in range(1, L):
        W = weight_params['W' + str(l)]
        b = weight_params['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = tanh(Z)
        cache_params['Z' + str(l)] = Z
        cache_params['A' + str(l)] = A
        A_prev = A

    # Forward propagation for output layer
    W = weight_params['W' + str(L)]
    b = weight_params['b' + str(L)]
    Z = np.dot(W, A_prev) + b
    A = softmax(Z)
    cache_params['Z' + str(L)] = Z
    cache_params['A' + str(L)] = A

    return cache_params


def compute_cost(Y, A):
    '''Computes cost of for the forward propagation - used with binary classification
    cost = logistic loss

    Parameters:
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


def compute_cost_softmax(Y, A, weight_params, lambd):
    ''' Computes cost with softmax - used with multiclass classification on last layer
    NOTE: Hardcoded to work only with 1 hidded layer
    '''
    m = Y.shape[1]
    #print('Y shape:')
    #print(Y.shape)
    #print('A shape:')
    #print(A.shape)

    log_calc = -np.sum(np.multiply(np.log(A), Y), axis=0)
    non_regularized_cost = 1/m * np.sum(log_calc)

    W1 = weight_params['W1']
    W2 = weight_params['W2']
    l2_regularization_cost = 1/m * lambd/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    
    cost = non_regularized_cost + l2_regularization_cost
    assert(isinstance(cost, float))

    return cost


def compute_cost_softmax_deep(Y, A, weight_params, lambd):
    ''' Computes cost with softmax - used with multiclass classification on last layer
    NOTE: This version works with all notwork configurations, set on config-file
    TODO: Implement
    '''
    m = Y.shape[1]
    #print('Y shape:')
    #print(Y.shape)
    #print('A shape:')
    #print(A.shape)

    log_calc = -np.sum(np.multiply(np.log(A), Y), axis=0)
    non_regularized_cost = 1/m * np.sum(log_calc)

    ''' regularization to be updated with many hidden layers
    W1 = weight_params['W1']
    W2 = weight_params['W2']
    l2_regularization_cost = 1/m * lambd/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    
    cost = non_regularized_cost + l2_regularization_cost
    assert(isinstance(cost, float))

    return cost
    '''
    return non_regularized_cost


def backward_propagation(X, Y, weight_params, cache_params, lambd):
    '''Backward propagation using gradient descent. 
       Computes delta between true values and computed weighted values
       NOTE: Hardcoded to work only with 1 hidded layer

    Parameters:
        X -- input parameters (images)
        Y -- true labels
        weight_params -- weight parameters
        cache_params -- Z, A, parameters computed during forward propagation
        lambd -- TODO define this

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
    dW2 = 1/m * np.dot(dZ2, A1.T) + lambd / m * W2
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(A1)   # TODO: check whether this is correct - should be _activate(Z1)?
    dW1 = 1/m * np.dot(dZ1, X.T) + lambd / m * W1
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

    '''
    print('dZ2 shape: ' + str(dZ2.shape))
    print('dW2 shape: ' + str(dW2.shape))
    print('db2 shape: ' + str(db2.shape))
    print('dZ1 shape: ' + str(dZ1.shape))
    print('dW1 shape: ' + str(dW1.shape))
    print('db1 shape: ' + str(db1.shape))
    exit(0)
    '''

    gradient_params = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    return gradient_params


def backward_propagation_deep(X, Y, weight_params, cache_params, lambd):
    '''Backward propagation using gradient descent. 
       Computes delta between true values and computed weighted values
       NOTE: This version works with all notwork configurations, set on config-file

    Parameters:
        X -- input parameters (images)
        Y -- true labels
        weight_params -- weight parameters
        cache_params -- Z, A, parameters computed during forward propagation
        lambd -- TODO define this

    Returns:
        gradient_params --  parameters of the gradients (weight - derivative)
    '''
    gradient_params = {}
    m = X.shape[1]
    L = len(weight_params) // 2  # number of layers in neural net, minus one
    
    # Calculate derivatives
    #TODO: In Coursera's course of "Neural Networks and Deep Learning", last week's notebook says:
    # Use: dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    for l in reversed(range(1, L + 1)):
        W = weight_params['W' + str(l)]
        A = cache_params['A' + str(l)]

        if l > 1:
            A_prev = cache_params['A' + str(l - 1)]
        else:
            A_prev = X

        if l == L:
            dZ = A - Y
        else:
            '''
            print('dZ shape: ' + str(dZ.shape))
            print('W + 1 shape: ')
            print(weight_params['W2'].shape)
            print('W + 1 shape: ' + str(weight_params['W' + str(l+1)].shape))
            #print('A shape: ' + str(A.shape))
            '''
            dZ = np.dot(weight_params['W' + str(l + 1)].T, dZ) * tanh_derivative(A)
        
        dW = 1/m * np.dot(dZ, A_prev.T) + lambd / m * W
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        '''
        print('dZ shape: ' + str(dZ.shape))
        print('dW shape: ' + str(dW.shape))
        print('db shape: ' + str(db.shape))
        '''
        gradient_params['dW' + str(l)] = dW
        gradient_params['db' + str(l)] = db

    return gradient_params


def update_params(weight_params, gradient_params, learning_rate):
    '''Updates weight parameters from the gradient

    Parameters:
        weight_params -- dictionary of weight and bias parameters
        gradient_params -- dictionary of gradienst from backpropagation
        learning_rate -- learning rate for parameter updates

    Returns:
        weight_params -- updated weight and bias parameters
    '''
    # update all parameters in place:
    # Wl = Wl + learning_rate * dWl ... and same for bl.
    for l in range(1, len(weight_params) // 2 + 1):
        weight_params['W' + str(l)] = weight_params['W' + str(l)] - gradient_params['dW' + str(l)] * learning_rate
        weight_params['b' + str(l)] = weight_params['b' + str(l)] - gradient_params['db' + str(l)] * learning_rate

    return weight_params


def run_model(X, Y, weight_params, iterations, learning_rate, lambd, classification_type):
    print('Cost:')
    cost = 0
    costs = []

    for i in range(iterations):
        #cache_params = forward_propagation(X, weight_params, classification_type)
        cache_params = forward_propagation_deep(X, weight_params)

        if classification_type == Classification.BINARY:
            cost = compute_cost(Y, cache_params['A2'])
        else:
            cost = compute_cost_softmax_deep(Y, cache_params['A2'], weight_params, lambd)

        gradient_params = backward_propagation_deep(X, Y, weight_params, cache_params, lambd)
        weight_params = update_params(weight_params, gradient_params, learning_rate)

        if __debug__:
            print_cadence = 100
        else:
            print_cadence = 10
        if i % print_cadence == 0:
            print('%.8f' % cost)
        
        costs.append(cost)
    
    if __debug__:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Cost diagram")
        plt.show()
    
    return weight_params


def predict(X, weight_params, classification_type):
    '''
    '''
    cache = forward_propagation(X, weight_params, classification_type)
    A2 = cache['A2']

    if classification_type == Classification.BINARY:
        # Binary classification: if value > 0.5, this is considered to be the match -> set to 1
        predictions = np.round(A2)
    else:
        # Multiclass classification: the biggest number in each 'softmax-column' is the most likely match
        # -> here predictions will be n x m array, each column being one-hot vector
        predictions = np.argmax(A2, axis=0)
        predictions.shape = (1, predictions.size)
        predictions = convert_to_one_hot(predictions)

    return predictions


def check_accuracy(Y, predictions):
    '''Checks the accuracy of trained network
    '''
    # TODO: Check whether this can be done better with numpy.compare or such
    #print('check_accuracy - Y shape')
    #print(Y.shape)
    correct_prediction = 0
    m = Y.shape[1]

    if Y.shape[0] == 1: # Binary classification
        print('Binary Classification')
        for i in range(m):
            # Notice that % indicates not only true positives, but also true negatives
            if Y[0,i] == predictions[0,i]:
                correct_prediction += 1
    else: # Multiclass classification
        print('Multiclass Classification')
        for i in range(m):
            if np.array_equal(Y[:,i], predictions[:,i]):
                correct_prediction += 1
        
    accuracy = correct_prediction / m *100
    #print('total samples: ' + str(m))
    #print('correct predictions: ' + str(correct_prediction))
    print('%: ' + str(accuracy))

    return accuracy


def predict_single(X, weight_params):
    '''
    '''
    cache = forward_propagation(X, weight_params)
    A2 = cache['A2']
    print('predict_single - A2')
    print(A2)
    prediction = np.argmax(A2, axis=0)
    propability = A2[prediction]
    
    return prediction, propability