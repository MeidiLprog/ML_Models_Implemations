import random
import numpy as np
import matplotlib.pyplot as plt

'''

backtracking
    base case

    for
    constraint

    add
    backtracking
    pop


'''

def find_empty_cell(p):
     for i in range(9):
          for j in range(9):
            if p[i][j] == 0:
                return i,j
     return None

def is_valid(p,l,c,n):

    if n in p[l,:]:
        return False
    if n in p[:,c]:
        return False
    
    bl = (l//3)*3
    bc = (c//3)*3

    if n in p[bl:bl+3,bc:bc+3]:
        return False

    return True

def sudoku(p):
     
    def backtracking():
        #base_case
        empty = find_empty_cell(p)
        if empty is None:
            return True
        l, c = empty 
        
        #constraint
        for n in range(1, 10):

            if not is_valid(p, l, c, n):
                continue
            #make choice
            p[l][c] = n
            #backtracking
            if backtracking():
                return True
            #unmake choice
            p[l][c] = 0   

        return False  

    backtracking()
    return p




grid = np.array([
    [5,3,0, 0,7,0, 0,0,0],
    [6,0,0, 1,9,5, 0,0,0],
    [0,9,8, 0,0,0, 0,6,0],

    [8,0,0, 0,6,0, 0,0,3],
    [4,0,0, 8,0,3, 0,0,1],
    [7,0,0, 0,2,0, 0,0,6],

    [0,6,0, 0,0,0, 2,8,0],
    [0,0,0, 4,1,9, 0,0,5],
    [0,0,0, 0,0,0, 0,7,9],
])


solved = sudoku(grid)
print(solved)








