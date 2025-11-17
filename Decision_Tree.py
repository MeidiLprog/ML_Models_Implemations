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
        
    #Important to remember, the loss is a a function to minimise
    def Loss(self,YLEFT,YRIGHT):
        #when split I need to retrieve nb_observations from left and right side in the node ex nb yes and nb_ no
        n_left = len(YLEFT)
        n_right = len(YRIGHT)
        #Calculating the sum of it for the loss + check of zero
        total = int(n_left) + int(n_right)
        if total == 0:
            raise ValueError("Dividing by zero is not allowed ! \n")
        #calculating the loss
        _LEFT = (n_left/total) * self.Gini(YLEFT)
        _RIGHT = (n_right/total) * self.Gini(YRIGHT)
    
        return _LEFT + _RIGHT


    def build_tree(self,X,y):
        pass

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



    