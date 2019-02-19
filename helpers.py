'''
Misc helper functions
'''

import numpy as np
import matplotlib.pyplot as plt

# TODO: Make this dynamic; get offset from the file itself
def read_data(filename, dataoffset):
    '''Read test- and training-sets (images, labels) from idx-file into numpy-array
     mnist datasets & spec: http://yann.lecun.com/exdb/mnist/
     -images is 28x28px size, each pixel valued 0-255
     -label is a single number, valued 0-9

     Args:
     filename -- name of the binary datafile
     dataoffset -- offset, after which the data payload in file starts

     Returns:
     numpy-array of payload
     '''

    f = open(filename, 'rb')
    f.seek(dataoffset)
    return np.fromfile(f, dtype = np.dtype(np.uint8))


def show_number(number_array, label_array, index):
    """ Visualizes the number and prints its label

    Args:
    number_array -- array of numbers
    label_array -- array of labels for numbers
    index -- index of number in array
    """

    num = number_array[:,[index]]
    num.shape = (28, 28)
    print(label_array[0, index])

    plt.imshow(num, cmap='gray')
    plt.show()


def get_data():
    ''' reads data from files and returns in suitable format

    Return:
    X       -- learning set of data
    X_test  -- test set of data
    Y       -- labels of learning set
    Y_test  -- labels if test set

    '''
    
    X = read_data('train-images.idx3-ubyte', 16)
    Y = read_data('train-labels.idx1-ubyte', 8)
    X_test = read_data('t10k-images.idx3-ubyte', 16)
    Y_test = read_data('t10k-labels.idx1-ubyte', 8)

    # normalize the data 0..1
    X = X/255
    X_test = X_test/255

    # data is arranged in n x m arrays (columns x rows)
    #   n = dimensions of single sample. For example in 28x28 image this equals 784
    #   m = number of samples
    X.shape = (60000, 784)
    X_test.shape = (10000, 784)
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

    return X, X_test, Y, Y_test