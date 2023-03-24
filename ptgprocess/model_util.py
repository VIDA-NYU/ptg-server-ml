import collections
import torch

class TensorQueue:
    def __init__(self, size, dim=0):
        self.queue = collections.deque(maxlen=size)
        self.dim = dim

    def clear(self):
        self.queue.clear()

    def push(self, x):
        self.queue.append(x)

    def tensor(self):
        return torch.stack(list(self.queue), dim=self.dim)
