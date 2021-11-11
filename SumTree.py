import numpy as np
import torch

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    # update to the root node
    def _propagate(self, idx, change):
        parent = torch.div((idx - 1), 2, rounding_mode='floor')

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, priority_idx, s):
        left = 2 * priority_idx + 1
        right = left + 1

        if left >= len(self.tree):
            return priority_idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, priority, data_idx):
        self.update(data_idx, priority)

    # update priority
    def update(self, data_idx, priority):
        priority_idx = data_idx + self.capacity - 1
        change = priority - self.tree[priority_idx]

        self.tree[priority_idx] = priority
        self._propagate(priority_idx, change)

    # get priority and sample
    def get(self, s):
        priority_idx = self._retrieve(0, s)
        data_idx = priority_idx - self.capacity + 1
 
        return self.tree[priority_idx], data_idx

    # get max priority
    def max(self):
        return np.max(self.tree[- self.capacity:])
    
    # get min priority
    def min(self, size):
        return np.min(self.tree[self.capacity - 1:self.capacity - 1 + size])    
