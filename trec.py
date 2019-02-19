import numpy as np
import helpers
import mlfunc
import time
import pickle
import matplotlib.pyplot as plt


'''
TODO:
-implement softmax for model to work with multiclass-classification
-fix relU functions - do not work currently
-implement regularization for cost function
-remove hardcodings from data read & array sizes
'''

X, X_test, Y, Y_test = helpers.get_data()

weight_params = mlfunc.init_params(X, Y)
#print("weight params /1: ")
#print(weight_params)

# Either:
# 1. train model to get new weight parameters
start_time = time.time()
weight_params = mlfunc.run_model(X, Y, weight_params, 100)
end_time = time.time()

# Or:
# 2. load weight parameters from earlier learned set
'''
with open('weight_params.pkl', 'rb') as f:
    weight_params = pickle.load(f)
'''

#print("weight params: /2 ")
#print(weight_params)
#mlfunc.check_accuracy(X, Y, weight_params)
predictions = mlfunc.predict(X, weight_params)
mlfunc.check_accuracy(Y, predictions)
print("time elapsed: " + str(end_time - start_time))
#print("predictions mean = " + str(np.mean(predictions)))

print("TEST SET")
predictions = mlfunc.predict(X_test, weight_params)
mlfunc.check_accuracy(Y_test, predictions)

# write learned weight parameters into disk
with open('weight_params.pkl', 'wb') as f:
    pickle.dump(weight_params, f)
