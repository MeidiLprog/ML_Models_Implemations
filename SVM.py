#IMPLEMENTATION OF SVM
 
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




class SVM:
    def __init__(self,kernel,C):
        if not isinstance(kernel,str):
            raise ValueError("Not a string")
        if not kernel in ["rbf","linear"]:
            raise ValueError("Kernel must be of type str and set the value of rbf or linear")
        if C < 0:
            raise ValueError("C is not to be negative")

        self.kernel = kernel
        if kernel == "rbf":
            self.C = np.inf
        if kernel == "linear":
            self.C = float(C)

        self.alpha = None
        self.X = None
        self.y = None
        self.K_function = None

    def kernel_calculation(self,X : np.ndarray):
        #we are to pick what kind of kernel we use, however we first need to check out a few prerequisites
        if not isinstance(X,np.ndarray):
            raise TypeError("X must be a numpy ndarray")
        if X.ndim != 2:
            raise ValueError("X must be a 2d matrix")
        if X.size == 0:
            raise ValueError("X cannot be an empty matrix, doesnt make any sense !")
        
        n = len(X)
        Ker = np.zeros((n,n))
        VARX = np.var(X)
        if VARX == 0:
            raise ValueError("Var cannot equate 0 \n")
        
        gamma = 1/(2* VARX)
        for i in range(len(X)):
            for j in range(len(X)):
                if self.kernel == "linear":
                    Ker[i,j] = np.dot(X[i],X[j])  
                else:
                    Ker[i,j] = np.exp( - gamma * (np.linalg.norm(X[i] - X[j])**2) )
        self.K_function = Ker
        return self.K_function

    def fit(self,X,y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Matrices aren't to be empty")
        self.X = X
        self.y = y.astype(float)
        print(f"Value {self.y} \n") 
        obs,variables = X.shape
        print(f"{obs} available\n")
        print(f"{variables} available\n")
        
        #we build our alpha as it is what we are looking for

        alpha = cp.Variable(obs)

        #now we buil the function to maximize

        K = self.kernel_calculation(X)
        K = (K + K.T) / 2
        #here thanks to quad_form we understand the relation between points and weights alpha i and x i
        #but before since I ran into issues with cp I have to check whether K is SPD so check whether the lower lambda is > 0
        eigvals = np.linalg.eigvalsh(K)
        print(f"Lowest eigen value {eigvals.min()}\n")
        if eigvals.min() < 0:
            print("The lowest eigen value is < 0 that means we must have a matrix with eigen values >0 to get a DSP ")
            eig_v, eig_vec = np.linalg.eigh(K)
            eig_v[eig_v < 0] == 0 #now we cheat a little bit to force our matrix to be DSP
            K = eig_vec @ np.diag(eig_v) @ eig_vec.T       
            
        K = cp.psd_wrap(K)
        z = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(cp.multiply(y,alpha),K))
        
        #we add our constraints sum alpha*i *yi == 0 and we limit C to indicate whether or not we penalize our constraints
        constraints = [cp.sum(cp.multiply(alpha,y)) == 0, alpha >= 0, alpha <= self.C]

  
        prob = cp.Problem(z, constraints)
        prob.solve(solver=cp.SCS, verbose=True)

        print(f" dual variable {constraints[0].dual_value} \n")
        print(f" dual variable {constraints[1].dual_value} \n")
        self.alpha = alpha.value
        return self.alpha

    #here we use our function "sign" to predict the class if f(x) > 0 : +1 else -1
    def predict(self,X):
        #predict the class of the observation
        n = X.shape[0]

        

        fx 

        return np.sign(fx)





from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, precision_score, accuracy_score


if __name__ == '__main__':
    load = load_iris()
    df = pd.DataFrame(data=load.data,columns=load.feature_names)
    df.describe()
    df.head(5)
    X = load.data[load.target != 2]
    y = load.target[load.target != 2]
    y = np.where(y == 0, -1,1)

    model = SVM(kernel="linear",C=1.0)
    try:
        model.fit(X,y)
    except ValueError:
        print("Values passed to the model are not correct \n")