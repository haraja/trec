'''
General helper functions
'''

import numpy as np
import matplotlib.pyplot as plt
import mlfunc
from enums import Classification
from PIL import Image, ImageOps
import PIL 
import os
import gzip


def read_mnist(filename, dataoffset):
    '''Read test- and training-sets (images, labels) from idx-file into numpy-array
     Data-files are expected to be .gz files in 'data'-folder
     mnist datasets & spec: http://yann.lecun.com/exdb/mnist/
     -images is 28x28px size, each pixel valued 0-255
     -label is a single number, valued 0-9

    Args:
        filename -- name of the binary datafile
        dataoffset -- offset, after which the payload data in file starts

    Returns:
        numpy-array of payload
    '''
    file_path = os.path.join('data', filename)
    with gzip.open(file_path, 'rb') as f:
        read_file = np.frombuffer(f.read(), dtype=np.uint8, offset=dataoffset)

    return read_file


def show_number(number_array, label_array, index):
    ''' Visualizes the number and prints its label

    Args:
        number_array -- array of numbers
        label_array -- array of labels for numbers
        index -- index of number in array
    '''
    num = number_array[:,[index]]
    num.shape = (28, 28)
    print(label_array[0, index])

    plt.imshow(num, cmap='gray')
    plt.show()


def mnist_to_array():
    ''' reads data from mnist-files and returns in suitable numpy array

    Return:
    X       -- learning set data
    X_test  -- test set data
    Y       -- learning set labels
    Y_test  -- test set labels

    Returns:
        X       -- learning set data
        X_test  -- test set data
        Y       -- learning set labels
        Y_test  -- test set labels
    '''
    X = read_mnist('train-images-idx3-ubyte.gz', 16)
    Y = read_mnist('train-labels-idx1-ubyte.gz', 8)
    X_test = read_mnist('t10k-images-idx3-ubyte.gz', 16)
    Y_test = read_mnist('t10k-labels-idx1-ubyte.gz', 8)

    # normalize the data 0..1
    X = X/255
    X_test = X_test/255

    # data is arranged in n x m arrays (columns x rows)
    #   n = dimensions of single sample. For example in 28x28 image this equals 784
    #   m = number of samples. Can be easily checked by getting munber of elements in labels-array
    X.shape = (Y.size, int(X.size/Y.size))
    X_test.shape = (Y_test.size, int(X.size/Y.size))
    X = X.T
    X_test = X_test.T
    # labels are arranged in row vectors
    Y.shape = (1, Y.size)
    Y_test.shape = (1, Y_test.size)

    # shape the output from labels to arrays
    Y = mlfunc.convert_to_one_hot(Y)
    Y_test = mlfunc.convert_to_one_hot(Y_test)

    #show_number(X, Y, np.random.randint(0, Y.size))

    return X, X_test, Y, Y_test


def jpg_to_array(file_name):
    ''' reads data from mnist-files and returns in suitable numpy array
    '''
    with Image.open(file_name) as image:
        #image_transformed = image.transform((28, 28), PIL.Image.AFFINE)
        image = image.resize((28,28))
        image = image.convert('L') # converts to 8-bit black and white
        image_invert = ImageOps.invert(image)
        image_array = np.fromstring(image_invert.tobytes(), dtype=np.uint8)
        image_array = tune_image(image_array)
        image_array.shape = (image_array.size, 1)       
    return image_array


def tune_image(image_array):
    ''' tune the image to be better fit for evaluation
    '''
    # initial very simple implementation. Removes background noise, but loses part of details as well
    image_array[image_array < 100] = 0
    return image_array