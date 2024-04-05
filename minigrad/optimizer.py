from abc import abstractmethod
from typing import List

from minigrad.core import Value


class Optimizer:

    parameters: List[Value]

    def __init__(self, parameters: List[Value]):
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
            p.grad_fn = None

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, parameters: List[Value], lr: float):
        super().__init__(parameters)

        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.v -= p.grad * self.lr
