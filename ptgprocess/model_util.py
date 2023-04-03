import collections
import torch

class TensorQueue(collections.deque):
    def __init__(self, size, dim=0):
        super().__init__(maxlen=size)
        self.dim = dim

    def append(self, x):
        super().append(x)
        return self

    def push(self, x):
        return self.append(x)

    def tensor(self):
        return torch.stack(list(self), dim=self.dim)
