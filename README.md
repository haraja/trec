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

Notice, that files need to be extracted and placed in same folder with py-files

## Status
This project is work in progress. Some implementation details:
-Only one hidden layer is initially supported
-Cost is minimized with gradient descent algorithm
