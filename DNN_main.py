# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:34:56 2020

@author: Rajiv

This program trains a Neural Network on the Bachelier formula
TO DO: add a regularisation function to the weights 

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from DNN_DL import NeuralNetwork
from DNN_Helper import writeOutput


# data - you;ll have to change this path of course depending on where you put your data
df = pd.read_excel("C:/Users/Rajiv/Google Drive/Science/Python/sandbox/DNN/Regression/BachelierDL/Generator.xlsx",usecols="M:U")

#extract the training sample
X, Y = df.iloc[:, [0,1,2,3]].values, df.iloc[:, [4,5,6]].values

# split into training batch and test batch
# the training batch is used to train the neural network and the test batch is used to test the trained network
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle = False, random_state=0)

# scaling for X and Y
# stdscX = StandardScaler()
stdscX = MinMaxScaler()
X_train_std = stdscX.fit_transform(X_train)
X_test_std = stdscX.transform(X_test)

stdscY = MinMaxScaler()
Y_train_std = stdscY.fit_transform(Y_train[:,[0]])
Y_test_std = stdscY.transform(Y_test[:,[0]])
        
# Now create the basic structure of the Neural Network, essentially the number of nodes at each layer. Size of N is the number of layers
N = np.array([4,20,20,1]) #number of nodes in each layer

# basic error checking on inputs. Should have more checks here I guess
if N[N.shape[0]-1] != Y_train[:,[0]].shape[1]:
    raise RuntimeError('Last layer must be equal to the number of class variables')
    
# this is the learning rate. The higher the rate, the faster it learns, but the more noisy and unstable the convergence is.
# higher learning rates can miss minima which is essentially what the NN is trying to find
eta = 2.0
L2 = 0#100.0 / X_train.shape[0]

# alpha parameter for *LU actiation functions - the below shouldn't work for elu but it does extremely well.
# const = stdscY.mean_[0] / np.sqrt(stdscY.var_[0])
# alpha = pow( 1.0 / const, 2)
alpha = 1.0

# now create the neural network class
NN = NeuralNetwork(N, L2, alpha)
NN.initialise('output_weights.csv')

# loss is a vector showing how the loss varies with each iteration of the algorithm (epoch)
loss = []

# we do the calculation in batches as it is more efficient
batch = 50
epochs = 2000

# fit the data
NN.fit(eta, L2, epochs, X_train_std, Y_train_std[:,[0]], batch, loss, 'output_weights.csv', 'output_diagnostics.csv')

# write the output to file
writeOutput(X_train, X_train_std, Y_train, Y_train_std, stdscX, \
            X_test, X_test_std, Y_test, Y_test_std, stdscY,     \
            NN, eta)



    













