#here's the the implementation of the logistic regression

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs,make_circles,make_moons


datasets = [
    make_moons(noise = 0.3, random_state = 0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 20), random_state=0)
]


def drawFunction(model,X,y,title="BOundaries") -> None:
    xmn, xmx = X[:,0].min() - 1, X[:,0].max() + 1
    ymn,ymx = X[:,1].min() -1, X[:,1].max() + 1

    xx,yy = np.meshgrid(np.linspace(xmn,xmx,200), np.linspace(ymn,ymx,200))

    grid = np.c_[xx.ravel(),yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    y_pred = model.predict(X)

    plt.figure(figsize=(8,12))
    plt.contour(xx,yy,Z,cmap="bwr",alpha=0.5)
    plt.scatter(X[:,0],X[:,1],c=y_pred,cmap="bwr",edgecolors="k")


    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()




    return 





def sigmoid(z : int):
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

#Trying out gradient's method
model = LogisticRegression(optimizer="gradient", iterations=200)
model.fit(X,y)

#Newton's one
modelnew = LogisticRegression(optimizer="newton", iterations=20) #newton's converging faster so no need to increase the number of iterations
modelnew.fit(X,y)

drawFunction(model,X,y,title="LR with Gradient")
plt.plot(range(len(model.Loss_cost)),model.Loss_cost)
plt.xlabel("Iterations")
plt.ylabel("Loss (log scale)")
plt.show()

drawFunction(modelnew,X,y,title="LR with Newton")
#I took the liberty of using plt.semilogy to obtain a better view of newton's convergence algorithm using a logarithmic scale
plt.semilogy(range(len(modelnew.Loss_cost)),modelnew.Loss_cost)
plt.xlabel("Iterations")
plt.ylabel("Loss (log scale)")
plt.show()




