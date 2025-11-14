import random

#backtracking problem

'''
Backtracking: Mind approach ? a 

Choices: All numbers in nums
Constraint: can't reuse the same number more than once in our path
Base Case: When the path length equals len(nums) which makes the first branch completed
Backtrack step : I do pop the last number added
Finally: I use backtracking([]) to start construction my list

Backtracking Template:
def backtrack(params):
    if base_case_condition:
        save_result
        return

    for choice in choices:
        if violates_constraints:
            continue

        make_choice
        backtrack(updated_params)
        undo_choice  # Backtracking Step
'''


class Permut:
    
    def permute(self,nums):
        result = []

        def backtracking(path):
        
            if len(path) == len(nums):
                result.append(path[:])
                return

            for n in nums:
                if n in path: continue
                path.append(n)
                backtracking(path)
                path.pop()
        backtracking([])
        return result


'''
Permutation problems

'''

# Test
perms = Permut()

nums = [1,2,3]

print("Backtracking done\n")
print()
print(f"Done : {perms.permute(nums)}")


