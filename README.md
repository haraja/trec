# TReg
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

Notice, that files need to be extracted and placed in same folder with py-files

## Status
This project is not finished, and not functional yet. Some implementation details>
-Only one hidden layer is initially supported
-Cost is minimized with gradient descent algorithm
-Sigmoid activation function is currently used in all layers & nodes.
 there is implementation for tanh as well

## Ideas during development
Initially task is to implement machine learning algorithm, using
deep neural network, to learn characters


Next:
-start to use tanh/relU instead of sigmoid function
-split "test-set" to dev- & test-set
