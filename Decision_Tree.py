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

        left_plus_right_total, total = np.unique(y, return_counts=True)
        

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



    