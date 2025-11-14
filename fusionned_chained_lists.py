import random
import numpy


#problem 2 -> to fusion sorted linked list

class Node:
    def __init__(self, val : int):
        if not isinstance(val,int):
             raise ValueError("Val must be of type int here \n")
        self._val = val
        self._next = None
    @property
    def val(self):
        return self._val
    @property
    def next(self):
        return self._next
    
    @val.setter
    def val(self,new_value):
        if not isinstance(new_value,int):
            raise ValueError("Must be an int")
        self._val = new_value
        

def create_chained_list(list) -> list:
    head = None
    for val in reversed(list):
        node = Node(val)
        node.next = head
        head = node

    return head

def print_chained_list(head):
    current = head
    while current is not None:
        print(f"Value in node : {current.val}", end="->")
        current = current.next
    print("None")
    return

'''
I use a mergin algorithm, since lists are already sorted out
I dont need to sort anything, therefore I create a function that merge two list
and repeat the process until there is no list left to be merged with another
and I return the process
we have n list, and the cost is n log(k) as we /2 the number of list to merge every time
'''

def merge_two_list(l1,l2):
    dummy = Node(0)
    current = dummy

    while l1 and l2: #here we use and as l1 and l2 are nodes and not list
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    if l1 is not None:
        current.next = l1
    else:
        current.next = l2
    return dummy.next






