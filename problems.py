import random
import numpy
import pandas as pd
import matplotlib.pyplot as plt


#first problem -> Knapsack

def knapsack_problem(obj_importance : list, obj_size : list, space_available : int):

    #init of our variables that are to be used
    obj_value : int = len(obj_importance)
    obj_taille : int = len(obj_size)
    sp = space_available
    
    #Creation of a list to handle what object is to be taken
    obj_taken = []
    for _ in range(obj_taille):
        obj_taken.append([False] * (space_available + 1))

    val_max = [0] * (sp + 1)

    for i in range(obj_value):
        for j in range(sp,obj_size[i],-1):
            #if we take an object i, we calculate the new value
            if val_max[j] != max(val_max[j], val_max[j - obj_size[i]] + obj_importance[i]):
                obj_taken[i][j] = True
            val_max[j] = max(val_max[j], val_max[j - obj_size[i]] + obj_importance[i])
    
    #now that we have done the first part, we move on to the choice of our objects
    index_of_obj = []
    t = sp
    for i in range(obj_value - 1, -1, -1):
        if obj_taken[i][t]:
            index_of_obj.append(i)
            t -= obj_size[i]

    index_of_obj.reverse()
    return index_of_obj, val_max[sp]

obj_importance = [60, 100, 120, 80, 30]
obj_size = [10, 20, 30, 15, 5]
space_available = 50


objc_taken, max_value = knapsack_problem(obj_importance, obj_size, space_available)
print("List of taken objects : ", objc_taken)
print("Max Value : ", max_value)









