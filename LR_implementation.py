#here's the the implementation of the logistic regression

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.datasets import make_moons,make_circles,make_blobs

datasets = [
    make_moons(noise = 0.3, random_state = 0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 20), random_state=0)
]

def sigmoid(z : int):
    func = 1/(1+ np.exp(-z))
    return func

class LogisticRegression:
    def __init__(self, optimizer="gradient", iterations=200):
        self.optimize = optimizer
        self.iterations = iterations
        
        self.beta = None
        self.Loss_cost = []

    def Loss(self, X,y,beta):
        p = sigmoid(X @ beta)
        #I added an epsilone to prevent log(0) otherwise it's gonna be infinite and the program shall crash
        eps = 1e-12
        cost = -(1/len(y))* np.sum(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
        return cost

    def fit(X,y):
        pass

    def predict(X):
        pass

