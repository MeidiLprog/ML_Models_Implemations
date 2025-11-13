import random
import numpy
import pandas as pd
import matplotlib.pyplot as plt


#first problem -> Knapsack

def knapsack_problem(obj_importance : list, obj_size : list, space_available : int):

    #init of our variables that are to be used
    obj_imp : int = len(obj_importance)
    obj_taille : int = len(obj_size)
    sp = space_available
    
    #Creation of a list to handle what object is to be taken
    obj_taken = [[False] * (space_available + 1) for _ in range(obj_taille)]

                




    return

