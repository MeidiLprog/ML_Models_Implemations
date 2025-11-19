import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

#Implementation of a decision tree using CART
# gini + tropy shall be used here


class Node:
    def __init__(self,feature = None,threshold = None,left = None,right = None,class_ = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_ = class_
        
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

    #in this function, I calculate my thresholds to perform my splitting based on the loss result later on
    def th_calculator(self,element_j : int):
        #I retrieve a column and sort out the values
        val = sorted(set(element_j))
        results = []

        if len(val) <= 1:
            return []
        else:
            for i in range(len(val) - 1 ):
                a = val[i]
                b = val[i+1]
                res = (a+b)/2
                results.append(res)

        return results


    def build_tree(self,X,y):
        
        if len(np.unique(y)) == 1:
            #I return a single class
            return Node(class_=y[0])
        
        #retrieve our feature, and th
        var_to_choose, th_to_choose = self.best_split(X,y)
        if var_to_choose is None:
            classes, counts = np.unique(y,return_counts=True)
            return Node(class_=classes[np.argmax(counts)])

        #retrieve the name
        chosen_var = X[:,var_to_choose]

        #check whether it's a categorial variable or a quantitative variable
        if self.isQuality(chosen_var) == True:
            l_in = (chosen_var == th_to_choose)
            l_right = (chosen_var != th_to_choose)
        else:
            l_in = (chosen_var <= th_to_choose)
            l_right = (chosen_var > th_to_choose)
            
        
        X_L,y_L = X[l_in], y[l_in]
        X_R,y_R = X[l_right], y[l_right]

        node_l = self.build_tree(X_L,y_L)
        node_r = self.build_tree(X_R,y_R)

        inode = Node(feature=var_to_choose,threshold=th_to_choose,left=node_l,right=node_r)
        return inode
    #difficult function to implement first case before going to sleep: Categorial variables
    def best_split(self,X,y):
        observations,variables = X.shape
        if observations == 0 or variables == 0:
            raise ValueError("neither rows nor columns are to be empty\n")
        
        max_feature = None
        threshold = None
        max_threshold = None
        best_loss_value = np.inf
        

        for j in range(variables):
            column = X[:,j]

            if self.isQuality(column) == False:

                #we calculate our threshold for a single quantitative variable

                threshold = self.th_calculator(column)
                for value in threshold:
                        
                        LEFT_INDEX = (column <= value)
                        RIGHT_INDEX = (column > value)
                        if LEFT_INDEX.sum() == 0 or RIGHT_INDEX.sum() == 0: continue
                        loss_value = self.loss(y[LEFT_INDEX],y[RIGHT_INDEX])
                        if loss_value < best_loss_value:
                            best_loss_value = loss_value
                            max_feature = j
                            max_threshold = value
            
            #case of a categorial variable
            else: 
                cat_var = np.unique(column)
                for c in cat_var:
                    

                    left_c = (column == c)
                    right_c = (column != c)

                    if left_c.sum() == 0 or right_c.sum() == 0: continue
                    loss_cat = self.loss(y[left_c],y[right_c])
                    if loss_cat < best_loss_value:
                        best_loss_value = loss_cat
                        max_feature = j
                        max_threshold = c
                        
        return max_feature,max_threshold


    def predict_one(self,X,node : Node):

        if node is None:
            node = self.root
        
        if node.class_ is not None:
            return node.class_
        
        val = X[node.feature]

        if self.isQuality(val) == True:
            if val == node.threshold:
                return self.predict_one(val,node.left)
            else:
                return self.predict_one(val,node.right)
        else:
            if val <= node.threshold:
                return self.predict_one(val,node.left)
            elif val > node.threshold:
                return self.predict_one(val,node.right)

    def Predict(self,X):
        pass


    def isQuality(self,column):
        if column.dtype == object or column.dtype.name == "category" or column.nuinique() < 10:
            return True
        else:
            return False



    