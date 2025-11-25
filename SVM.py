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

        self.alpha = None
        self.X = None
        self.y = None
        self.K_function = None

    def kernel_calculation(self,X : np.matrix,ker_nel = 0):
        #we are to pick what kind of kernel we use
        if ker_nel == 0: 
            for i in range(len(X)):
                for j in range(len(X)):
                    self.K_function = X[i] @ X[j]
        else:
            for i in range(len(X)):
                for j in range(len(X)):
                    

        return

    def fit(self,X,y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Matrices aren't to be empty")
        self.X = X
        self.y = y 
        obs,variables = X.Shape
        matrix = self.zeros(obs,obs)
        self.kernel_calculation(matrix,ker_nel=1)
        print(f"{obs} available\n")
        print(f"{variables} available\n")
        


        return


    def predict(self,X):
        pass


