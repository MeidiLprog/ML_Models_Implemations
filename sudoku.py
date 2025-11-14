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

def sudoku(puzzle):
   
     
    def backtracking():
        #base_case
        empty = find_empty_cell(puzzle)
        if empty is None:
            return True
        l, c = empty 
        
        #constraint
        for n in range(1, 10):

            if not is_valid(puzzle, l, c, n):
                continue
            #make choice
            puzzle[l][c] = n
            #backtracking
            if backtracking():
                return True
            #unmake choice
            puzzle[l][c] = 0   

        return False  

    backtracking()
    return puzzle




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


solved = sudoku(grid.copy())
print(solved)

def showSudoku(puzzle,puzzled_solved):
    
    plt.figure(figsize=(6,6)) #window size
    ax = plt.gca() #get current axe

    ax.imshow(np.ones((9,9)),cmap="gray_r") #display the matrix as an image and the gray_r just means reversed so white -> black

    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(10):
        lw = 3 if i%3 == 0 else 1
        ax.axhline(i - 0.5, color = 'black', linewidth=lw)
        ax.axvline(i - 0.5, color = 'black', linewidth=lw)

    
    for l in range(9):
        for c in range(9):
            if puzzle[l][c] != 0:
                color = 'black'
            else:
                color = 'blue'
            ax.text(c,l,puzzled_solved[l][c],
                    ha = 'center', va='center',
                    fontsize=12,color=color
                    
                    )

    plt.show()
showSudoku(grid,solved)