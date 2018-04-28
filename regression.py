from sympy import *
import math
import csv
import sys
import random

class NonlinearRegression:
    def __init__(self):
        self.inputs_list = []
        self.outputs_list = []
        
    def addMatrixRow(self, ins, out):
        self.inputs_list.append(ins)
        self.outputs_list.append(out)
        
    def linearRegress(self):
        x = Matrix(self.inputs_list)
        y = Matrix(self.outputs_list)
        xt = x.T
        xd = (xt * x).inv() * xt
        w = xd * y
        return w
        
    def testWeights(self, w):
        for i in range(0, len(self.inputs_list)):
            x = Matrix([self.inputs_list[i]])
            yhat = (x * w)[0,0]
            y = self.outputs_list[i][0]
            dif = (y-yhat)
            print(y, " " , yhat, " ", dif, " ", (dif / y))


        
    def computeDerivatives(self):
        args = list(self.coeffs)
        args += self.inputs
        args.append(self.output)
        self.args=tuple(args)
        
        e = self.errorf
        derivs = []
        for weight in self.coeffs:
            d = diff(e, weight)
            print('dE/d', weight, " = ", d)
            d = lambdify(args, d)
            derivs.append(d)

        self.derivs = derivs
        self.errorl = lambdify(args, e)

    
    def setup(self, output, inputs, formula, coeffs):
        print(output)
        print(inputs)
        self.formula = formula
        self.coeffs = coeffs
        self.inputs = inputs
        self.output = output
        self.errorf = ((output - formula)**2)/2
        print("E = ", self.errorf)
        self.computeDerivatives()
        
    def initialWeights(self, w):
        self.weights = w
    
    def sampleError(self, row): # Assuming output is last element
        args = self.weights + row
        return self.errorl(*args)
    
    def datasetError(self, dataset):
        te = 0.0
        for row in dataset:
            te += self.sampleError(row)
        return sqrt(te / len(dataset));
    
    def sampleTrain(self, row, lr):
        args = self.weights + row
        for j in range(0,len(self.derivs)):
            try:
                gradient = self.derivs[j](*args)
                self.weights[j] -= lr * gradient
            except:
                pass
    
    def datasetTrain(self, dataset, lr):
        for row in dataset:
            self.sampleTrain(row, lr)

