#here's the the implementation of the logistic regression

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons,make_circles,make_blobs

datasets = [
    make_moons(noise = 0.3, random_state = 0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 20), random_state=0)
]

def sigmoid(z : int):
    if not isinstance(z,int):
        raise TypeError("sigmoid retrieves integers")
    if z < 0:
        raise ValueError("cannot be less than zero")
    func = 1/(1+ np.exp(-z))
    return func

class LogisticRegression:
    def __init__(self, optimizer="gradient", iterations=200):
        if optimizer.lower() == "gradient" or optimizer.lower() == "newton":
            self.optimize = optimizer
        else:
            raise ValueError("gradient and newton are the only acceptable methods \n")
        
        self.iterations = iterations
        
        self.beta = None
        self.Loss_cost = []

    def Loss(self, X,y,beta):
        p = sigmoid(X @ beta)
        #I added an epsilone to prevent log(0) otherwise it's gonna be infinite and the program shall crash
        eps = 1e-9
        cost = -(1/len(y))* np.sum(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
        return cost

    def fit(self,X,y):
        
        if self.optimize == "gradient":
            #I need to add the biais to X beta0 + beta1 x1 + beta2 x2 ... beta n xn
            X = np.hstack([np.ones((X.shape[0],1)),X])

            #Retrieving Highst lambda value from SVD
            _, S, _ = np.linalg.svd(X,full_matrices=False)
            _highest_lambda = S[0]

            #here I initialize beta as I need to do a matrix multiplicate with X
            beta = np.zeros(X.shape[1])


            alpha = 4/(_highest_lambda**2)

            for i in range(self.iterations):
            
            #calculation of the gradient here sigmoid play the role of X*THETA in MSE, we just use the probalistic approach
                gradient = X.T @ (sigmoid(X @ beta) - y) 
                #calculation of parameters beta without momentum, applying a normal GD       
                beta = beta - alpha * gradient
            
                self.Loss_cost.append(self.Loss(X,y,beta)) # I save the beta to plot the convergence of it
                if np.linalg.norm(gradient) < 1e-6:
                    print("Local globalizer of f found \n")
                    break
            self.beta = beta
        
        elif self.optimize == "newton":

            X = np.hstack([np.ones((X.shape[0],1)),X])
            beta = np.zeros(X.shape[1])

            t = 0
            while t < self.iterations:
                p = sigmoid(X @ beta)
                gradient = X.T @ (p - y)
                W = np.diag(p * (1 - p))
                H = X.T @ W @ X
                delta = np.linalg.solve(H, gradient)
                beta = beta - delta
                self.Loss_cost.append(self.Loss(X, y, beta))

                if np.linalg.norm(gradient) < 1e-6:
                    break
                t += 1

            self.beta = beta




    def predict(self,X):
        
        X = np.hstack([np.ones((X.shape[0],1)),X])

        z = X @ self.beta

        p = sigmoid(z)
        return (p >= 0.5).astype(int)

import matplotlib.pyplot as plt


X , y = datasets[0]

model = LogisticRegression(optimizer="newton", iterations=200)
model.fit(X,y)


y_pred = model.predict(X)


plt.scatter(X[:,0], X[:,1],c=y_pred,cmap="bwr",alpha=0.7)
plt.title("Classification using logistic regression")
plt.show()

#display loss function

plt.plot(model.Loss_cost)
plt.title("Evolution of the loss function")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()
