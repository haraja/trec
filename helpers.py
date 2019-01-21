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
