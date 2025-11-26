#IMPLEMENTATION OF SVM
 
import CVXPY as cp
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
        gamma = 1/(2*np.var(X))
        if ker_nel == 0: 
            for i in range(len(X)):
                for j in range(len(X)):
                    self.K_function[i,j] = X[i] @ X[j]
        else:
            for i in range(len(X)):
                for j in range(len(X)):
                    self.K_function[i,j] = np.exp( -gamma * (np.linalg.norm(X[i] - X[j]**2)) )

        return self.K_function

    def fit(self,X,y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Matrices aren't to be empty")
        self.X = X
        self.y = y 
        obs,variables = X.Shape
        matrix = self.zeros(obs,obs)
        print(f"{obs} available\n")
        print(f"{variables} available\n")
        
        #we build our alpha as it is what we are looking for

        alpha = cp.Variable(obs)

        #now we buil the function to maximize

        K = self.kernel_calculation(matrix,ker_nel=1)
        z = cp.Maximize(np.sum(alpha) + cp.quad_form(cp.multiply(y,alpha),K))
        
        #we add our constraints sum alpha*i *yi == 0 and we limit C to indicate whether or not we penalize our constraints
        constraints = [alpha @ y == 0, alpha >= 0 and alpha <= self.C]


        prob = cp.Problem(z,constraints)
        prob.solve()

        print(f" dual variable {constraints[0].dual_value} \n")
        print(f" dual variable {constraints[1].dual_value} \n")
        print(f"Value: {constraints.value} \n")

        return


    def predict(self,X):
        pass


