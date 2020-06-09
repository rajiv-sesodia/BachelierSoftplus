# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:34:56 2020

@author: Rajiv

This program trains a Neural Network on the Bachelier formula
TO DO: add a regularisation function to the weights 

"""

import numpy as np
from ActivationFunctions import ActivationFunctions
from DNN_Helper import readWeightsAndBiases
from DNN_Helper import writeWeightsAndBiases
from DNN_Helper import writeDiagnostics

# One of my first Neural Networks
class NeuralNetwork:
    
    def __init__(self, N, L2 = 0, alpha = 0.01):
        # L is the number of layers in the neural network
        # N is an array containing the number of nodes in each layer
        self.L = N.shape[0]
        self.N = N
        self.af = ActivationFunctions('softplus', alpha)
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
        self.L2 = L2

        
    def initialise(self, weightsAndBiasesFile=''):
    
        # initialise weights and biases
        self.w = []
        self.b = []
        
        # if weights file supplied, read them in
        if weightsAndBiasesFile:
            self.w, self.b = readWeightsAndBiases(weightsAndBiasesFile)
            return
            
        # otherwise set randome weights
        # always important to set the seed for comparability
        np.random.seed(0)
        
        # these are members of the NN class
        self.w.append(np.zeros((self.N[0],self.N[0])))
        self.b.append(np.zeros(self.N[0]))
        
        # the initial weights and biases are set to a random number taken from a (standard) normal distribution, i.e.
        # with mean 0 and variance 1
        for l in range(1,self.L):
            # self.w.append(np.random.normal(0.0,1.0,(self.N[l-1],self.N[l])))
            # self.b.append(np.random.normal(0.0,1.0,self.N[l]))
            self.w.append(np.sqrt(2.0/self.N[l])*np.random.rand(self.N[l-1],self.N[l]))
            self.b.append(np.sqrt(2.0/self.N[l])*np.random.rand(self.N[l]))
            
               
    def feedForward(self, X):                
        
        # calculates the perceptron value (z) and activated value (a) through the activation function
        X = X if X.ndim > 1 else X.reshape(1,X.shape[0])
        
        # note, no activation for input layer, a = X
        a = [X]
        z=[np.zeros((X.shape[0],X.shape[1]))]
        for l in range(0,self.L-1):            
            z.append(a[l].dot(self.w[l+1]) + self.b[l+1])
            a.append(self.phi(z[l+1]))
            
        return z, a
    
    
    def calc_dcdz(self, a, z, y):
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        
        # special treatment for last layer as this derivative is from the cost function itself.
        # for the remainder of the layers following recursion formulae
        # dcdz = [(2.0 / self.N[self.L-1]) * np.multiply( (a[self.L-1]-y) , self.dphi(z[self.L-1])) ]
        m = a[self.L-1].shape[0]
        dcdz = [(2.0 / m) * np.multiply( (a[self.L-1]-y) , self.dphi(z[self.L-1])) ]
        dyda = [np.ones((m, self.N[self.L-1]))]
        for l in reversed(range(0,self.L-1)):
              dcdz.insert(0,np.multiply( dcdz[0].dot((self.w[l+1]).T), self.dphi(z[l])))
              dyda.insert(0, np.multiply(dyda[0], self.dphi(z[l+1])).dot(self.w[l+1].T))

        return dcdz, dyda

    # backpropogation of cost calculated at output layer through the network, updating the weights and biases as we go along
    # returns dcdz which is needed for derivative
    def backProp(self, eta, z, a, y):
                
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        dcdz, dyda = self.calc_dcdz(a, z, y)

        # calculating derivatives of the cost (c) function w.r.t weights (w), i.e. (dcdw) and bias (b), i.e. (dcdb) 
        # and updating weights MUST be done AFTER the derivatives are calculated,
        # as the latter depends on the former             
        for l in reversed(range(1,self.L)):
            dcdw = a[l-1].T.dot(dcdz[l]) + self.L2*2.0*self.w[l]
            dcdb = np.sum(dcdz[l], axis = 0)
            self.w[l] -= eta * dcdw
            self.b[l] -= eta * dcdb
        
    
    def calcSplitLoss(self, X, Y):
        m = Y.shape[0]
        zTemp, aTemp = self.feedForward(X)
        loss1 = np.sum(np.square(Y-aTemp[self.L-1])) / m
        
        loss2 = 0
        for l in range(self.L):
            loss2 += self.L2* np.sum(np.square(self.w[l])) / m

        return loss1, loss2
    
    
    def calcLoss(self, X, Y):
        m = Y.shape[0]
        zTemp, aTemp = self.feedForward(X)
        loss = np.sum(np.square(Y-aTemp[self.L-1])) / m
        
        for l in range(self.L):
            loss += self.L2* np.sum(np.square(self.w[l])) / m
            
        return loss
    
     
    def fit(self, eta, L2, epochs, X, Y, batchSize, loss, weightsAndBiasesFile='', diagnosticsFile=''):
        rgen = np.random.RandomState(1)
        
        for epoch in range(epochs):
            
            # shuffle
            r = rgen.permutation(len(Y))

            # loop over entire randomised set
            for n in range(0, len(Y) - batchSize + 1, batchSize):
                indices = r[n : n + batchSize]
                z, a = self.feedForward(X[indices])
                self.backProp(eta, z, a, Y[indices])                
        
            # check error
            if epoch % 100 == 0:
                loss1, loss2 = self.calcSplitLoss(X,Y)
                print('epoch = ', epoch, 'loss1 = ',loss1, 'loss2 = ', loss2, 'eta = ', eta)
                loss.append([epoch, loss1 + loss2])

        if weightsAndBiasesFile:
            writeWeightsAndBiases(self.w, self.b, weightsAndBiasesFile)
            
        if diagnosticsFile:
            error_w, error_b = self.GradientCheck(X,Y)
            error_a = self.feedForwardCheck(X[0])
            writeDiagnostics(error_w, error_b, error_a, self.w, self.b, loss, diagnosticsFile)

                
    def gradient(self, eta, X, Y, stdscX, stdscY):
        
        # derivative calculation
        z, a = self.feedForward(X)
        dcdz, derivatives_unscaled = self.calc_dcdz(a, z, Y)   
        
        # re-scale the output
        # derivatives = np.divide(derivatives_unscaled[0], np.sqrt(stdscX.var_))
        # if stdscY.with_std:
        #     derivatives = np.multiply(derivatives, np.sqrt(stdscY.var_))
        derivatives = np.divide(derivatives_unscaled[0], stdscY.data_range_ / stdscX.data_range_)
        
        return derivatives
    
    

    def backPropGradientCheck(self, eta, z, a, y):
                
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        dcdz, dyda = self.calc_dcdz(a, z, y)

        # calculating derivatives of the cost (c) function w.r.t weights (w), i.e. (dcdw) and bias (b), i.e. (dcdb) 
        # and updating weights MUST be done AFTER the derivatives are calculated,
        # as the latter depends on the former             
        dcdw = [a[self.L-2].T.dot(dcdz[self.L-1]) + self.L2*2.0*self.w[self.L-1]]
        dcdb = [np.sum(dcdz[self.L-1], axis = 0)]
        for l in reversed(range(1,self.L-1)):
            dcdw.insert(0, a[l-1].T.dot(dcdz[l]) + self.L2*2.0*self.w[l])
            dcdb.insert(0, np.sum(dcdz[l], axis = 0))
        
        dcdw.insert(0,0)
        dcdb.insert(0,0)
        
        return dcdw, dcdb
    


    def GradientCheck(self, X, Y):        
        
        # base case
        eps = 1e-06
        if self.w == []:
            self.initialise()
        
        # calculate derivative
        z, a = self.feedForward(X)
        dcdw, dcdb = self.backPropGradientCheck(1.0, z, a, Y)
        
        # calculate error in weights
        for n in range(1, self.L):
            for lm1 in range(self.N[n-1]):
                for l in range(self.N[n]):
                    
                    # up
                    self.w[n][lm1][l] += eps
                    C_up = self.calcLoss(X, Y)
                    
                    # down
                    self.w[n][lm1][l] -= 2.0*eps
                    C_down = self.calcLoss(X, Y)
                    
                    # error in deriv
                    dcdw[n][lm1][l] -= (C_up - C_down) / (2.0 * eps)
                    
                    # restore original value
                    self.w[n][lm1][l] += eps
                                      
     

        # calculate error in bias
        for n in range(1, self.L):    
            for l in range(self.N[n]):
                
                # up
                self.b[n][l] += eps
                C_up = self.calcLoss(X, Y)
                
                # down
                self.b[n][l] -= 2.0 * eps
                C_down = self.calcLoss(X, Y)
                
                # error in deriv
                dcdb[n][l] -= (C_up - C_down) / (2.0 * eps)
                
                # restore original value
                self.b[n][l] += eps
            
        return dcdw, dcdb
            
        
    def feedForwardCheck(self, X):         
        
        # temporary override of class members
        import copy
        w_ = copy.deepcopy(self.w)
        b_ = copy.deepcopy(self.b)
        name = self.af.name
        alpha = self.af.alpha
        self.af = ActivationFunctions('linear')
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
            
        for l in range(1,self.L):
            for i in range(self.N[l-1]):
                for j in range(self.N[l]):
                    self.w[l][i][j] = 1.0/(self.N[l-1])
                   
        for l in range(1,self.L):        
            for j in range(self.N[l]):
                self.b[l][j] = 0.0
            
        result = np.full(self.N[self.L-1], np.average(X))
        

        z, a = self.feedForward(X)
        error = result - a[self.L-1]
        
        # restore class members to original values
        self.w = copy.deepcopy(w_)
        self.b = copy.deepcopy(b_)
        self.af = ActivationFunctions(name, alpha)
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
        
        return error
        