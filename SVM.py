#IMPLEMENTATION OF SVM

import CVXPY    
import matplotlib.pyplot as plt
import numpy as np


class SVM:
    def __init__(self,kernel,C):
        if not isinstance(kernel,str):
            raise ValueError("Not a string")
        if not kernel in ["rbf","linear"]:
            raise ValueError("Kernel must be of type str and set the value of rbf or linear")
        if not isinstance(C,float) or not isinstance(C,int):
            raise ValueError("C must be an int or a float")

        self.kernel = kernel
        if kernel == "rbf":
            self.C = np.inf
        if kernel == "linear":
            self.C = 0.025

    def fit(self,X,y):
        pass

    def predict(self,X):
        pass


