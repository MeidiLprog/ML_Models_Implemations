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
        
        self.n_estimator = n_estimators
        self.criterion = criterion
        self.trees = []


    def fit(self,X,y):

        trees_to_store = []
        #Since we are using a bootstrap approach to train our forest, we are to use random samplings, therefore np.random.choice

        size_of_sample = len(X)
        
        i = 0
        while i < self.n_estimator:
            samp = np.random.choice(size_of_sample,size=size_of_sample,replace=True)
            
            #random samples for n lines, and p variables
            X_samp = X[samp]
            y_samp = y[samp]
            tree_built = Decision_Tree.Tree(criterion=self.criterion)
            tree_built.fit(X_samp,y_samp)
            trees_to_store.append(tree_built)
            i += 1
        self.trees = trees_to_store

    def predict(self,X):
        final_aggr = []
        train = np.array([tree_pred.predict(X) for tree_pred in self.trees]) # each tree trained
        for i in range(X.shape[0]): #for each individual
            values, counts = np.unique(train[:,i],return_counts=True) #for each individuals return the number of unique classes and also theirs counts, ex: [0,1] counts = [14,31]
            final_aggr.append(values[np.argmax(counts)]) # return the index of the highest count, therefore, the class majority
        return np.array(final_aggr) #return the final list of the predictions