import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

#Implementation of a decision tree using CART
# gini + tropy shall be used here






class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.class_ = None
        
class Tree:
    def __init__(self,criterion="gini"):
        if str(criterion).lower() not in ["gini","entropy"]:

            self.criterion = criterion
        else:
            self.criterion = criterion
        
        self.build_tree() = None
        self.root = None

    def fit(self,X,y):
        self.root = self.build_tree(X,y)

    def Gini(self,y):
        #left_plus_right_total was used to theoritically getting the classes, but I dont need it actually
        left_plus_right_total, total = np.unique(y, return_counts=True)
        parts : list = []
        #I divide each proporition by the total
        for i in total:
            parts.append((i)/(len(y)))

        #appplying the square 
        sum_sq = 0
        for j in range(len(parts)):
            sum_sq += (parts[j])**2

        #return Gini
        return 1 - sum_sq
        

    def Loss(self,YLEFT,YRIGHT):
        n_left = len(YLEFT)
        n_right = len(YRIGHT)
        
        total = int(n_left) + int(n_right)
        if total == 0:
            raise ValueError("Dividing by zero is not allowed ! \n")

        _LEFT = (n_left/total) * self.Gini(YLEFT)
        _RIGHT = (n_right/total) * self.Gini(YRIGHT)
    
        return _LEFT + _RIGHT


    def build_tree(X,y):
        pass

    def best_split(self):
        pass

    def Predict(self,X):
        pass

    def predict_one(self):
        pass

    def isQuality(self,column):
        if column.dtype == object or column.dtype.name == "category" or column.nuinique() < 10:
            return True
        else:
            return False



    