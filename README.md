# TRec
Program to recognize handwritten characters

## Why trec?
To practice implementing deep neural network, to be used in learning characters.
All algorithms are written from the scratch. Readily available libraries are
used primarily for matrix operations. Target is to work with big dataset, so
usage of for-loops is avoided and instead matrix-operations are preferred.

## Getting started
Needed data of the character-sets, which is used for learning, is from MNIST.
This data is not included in this Git project. You need to download the sets from:
http://yann.lecun.com/exdb/mnist/
...and place the .gz-files to data-folder


## Status
This project is work in progress. Some implementation details:
-Only one hidden layer is initially supported
-Cost is minimized with gradient descent algorithm

## Structure
-trec.py: this is the script which be run by you. Will call all other needed functions on other files. 
Parses the command line parameters, gets the data, runs the model
-helpers.py: general helper functions
-mlfunc.py: machine learning related functions

## Code notation
X: input array
Y: marks labels
training_set: st used to train the model (X, Y in code are training sets)
*_test: set used to evaluate correctness of trained model