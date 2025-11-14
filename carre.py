import random

#again, another problem of backtracking

def manhattan(p1 : tuple,p2 : tuple) -> int:
    
    d = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    return d

print(manhattan((2,3),(4,5)))


def max_min_distance(side, points, k):
    points = sorted(points)  
    
    lo, hi = 0, 2 * side
    ans = 0

    while lo <= hi:
        mid = (lo + hi) // 2

        if can(mid, points, k):
            ans = mid
            lo = mid + 1   
        else:
            hi = mid - 1   

    return ans



def can(D: int, points, k: int):

    selection = []

    def backtracking(start):
        # base case
        if len(selection) == k:
            return True

        for i in range(start, len(points)):
            p = points[i]

            # check Manhattan constraint
            ok = True
            for q in selection:
                if manhattan(p, q) < D:
                    ok = False
                    break
            if not ok:
                continue  # skip this point

            # make choice
            selection.append(p)

            # recurse
            if backtracking(i + 1):
                return True

            # undo choice
            selection.pop()

        return False

    return backtracking(0)


side = 2
points = [[0,0],[1,2],[2,0],[2,2],[2,1]]
k = 4

print(max_min_distance(side, points, k))   

    




