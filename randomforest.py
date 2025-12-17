import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Decision_Tree




class RandomForest:
    def __init__(self,n_estimators=10, criterion="gini"):
        if not isinstance(n_estimators,int):
            raise ValueError("Estimator indicates the number of trees, cannot be of another type than int")
        if not isinstance(criterion,str):
            raise ValueError("Criterion must be of type str")
        if criterion.lower() not in ["gini","entropy"]:
            raise ValueError("Only gini and entropy are available")
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.trees = []


    def fit(self,X,y):

        trees_to_store = []
        #Since we are using a bootstrap approach to train our forest, we are to use random samplings, therefore np.random.choice

        size_of_sample = len(X)
        variables = X.shape[1] #this variable is made as we're going to apply the feature bagging approach to optimize our trees

        i = 0
        while i < self.n_estimators:
            samp = np.random.choice(size_of_sample,size=size_of_sample,replace=True)
            
            #random samples for n lines, and p variables
            X_samp = X[samp]
            y_samp = y[samp]

            sub_variables = np.random.choice(np.arange(variables),size=int(np.sqrt(variables)),replace=False)
            X_samp = X_samp[:,sub_variables]
            tree_built = Decision_Tree.Tree(criterion=self.criterion)
            tree_built.fit(X_samp,y_samp)
            trees_to_store.append((tree_built,sub_variables))
            i += 1
        self.trees = trees_to_store 
        
    def predict(self,X):
        final_aggr = []
        train = np.array([tree_pred.predict(X[:,features]) for tree_pred,features in self.trees]) # each tree trained
        for i in range(X.shape[0]): #for each individual
            values, counts = np.unique(train[:,i],return_counts=True) #for each individuals return the number of unique classes and also theirs counts, ex: [0,1] counts = [14,31]
            final_aggr.append(values[np.argmax(counts)]) # return the index of the highest count, therefore, the class majority
        return np.array(final_aggr) #return the final list of the predictions





import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score,confusion_matrix
from sklearn.datasets import make_moons, make_circles, make_blobs
datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=100, centers=2, n_features=2,
               center_box=(0, 20), random_state=0)
]
def drawFunction(model,X,y,title="BOundaries") -> None:
    xmn, xmx = X[:,0].min() - 1, X[:,0].max() + 1
    ymn,ymx = X[:,1].min() -1, X[:,1].max() + 1

    xx,yy = np.meshgrid(np.linspace(xmn,xmx,200), np.linspace(ymn,ymx,200))

    grid = np.c_[xx.ravel(),yy.ravel()]
    Z = np.array(model.predict(grid)).reshape(xx.shape)


    y_pred = model.predict(X)

    plt.figure(figsize=(8,12))
    plt.contour(xx,yy,Z,cmap="bwr",alpha=0.5)
    plt.scatter(X[:,0],X[:,1],c=y_pred,cmap="bwr",edgecolors="k")


    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()

names = ["Moons", "Circles", "Blobs"]

if __name__ == "__main__":
    for i in range(len(datasets)):
        X, y = datasets[i]
        
        rf = RandomForest(n_estimators=30, criterion="gini")
        rf.fit(X, y)
        
        drawFunction(rf,X,y,title=f"Random Forest (Gini) on {names[i]}")
