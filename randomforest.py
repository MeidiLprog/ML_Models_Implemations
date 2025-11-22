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