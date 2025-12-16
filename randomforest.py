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


if __name__ == "__main__":
    iris =  load_iris()
    
    X = iris.data.astype(object) # my decision tree uses comparaisons
    y = iris.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    rf = RandomForest(n_estimators=5,criterion="gini")

    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test, y_pred)}\n")
    print(f"Precision : {precision_score(y_test, y_pred, average='macro')}\n")
    print(f"Recall : {recall_score(y_test, y_pred, average='macro')}\n")
    print(f"F1 score : {f1_score(y_test, y_pred, average='macro')}\n")
    print(f"Confusion Matrix :\n{confusion_matrix(y_test, y_pred)}\n")


#ADDING LATER PRECISION_CURVE to be able to measure how convergent is our algorithm
n_trees_list = [1, 5, 10, 15, 30, 50, 75, 100, 150]
accuracies = []

for n in n_trees_list:
    rf = RandomForest(n_estimators=n, criterion="gini")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"{n} tree -> accuracy = {acc:.3f}")

plt.figure(figsize=(8,5))
plt.plot(n_trees_list, accuracies, marker='o', color='green')
plt.xlabel("Nb trees")
plt.ylabel("(Accuracy)")
plt.title("Random forest convergence")
plt.grid(True)
plt.show()