from __future__ import annotations

import math
from abc import abstractmethod
from typing import Optional, List, Union, Set, Iterable, Any

ValueMatrix = List[List['Value']]


class Value:
    v: Union[int, float]
    grad: float
    grad_fn: Optional[GradFunction]

    def __init__(self, v: Union[int, float], grad: float = 0, grad_fn: Optional[GradFunction] = None):
        if not isinstance(v, (int, float)):
            raise ValueError(f'v must be of type int or float but is of type {type(v)} with value {v}')

        self.v = v
        self.grad = grad
        self.grad_fn = grad_fn

    def backward(self):
        if self.grad_fn is None:
            return

        self.grad = 1

        visited: Set[Value] = set()
        topsort: List[Value] = []

        def _traverse(v: Value):
            if v in visited or v.grad_fn is None:
                return

            visited.add(v)
            for child in v.grad_fn.inputs():
                _traverse(child)
            topsort.append(v)

        _traverse(self)
        # Check that gradients of all Values in the current computation graph (except self) are set to 0. If this is not the case,
        # we backpropagated through the same Value twice, which is not supported.
        for v in topsort[:-1]:
            if v.grad != 0:
                raise RuntimeError(f'Can not backpropagate multiple times through {v} with object_id={id(v)}')

        topsort.reverse()
        for v in topsort:
            # Note that at each iteration, v.grad contains its *final* value's gradient for this particular backward call,
            # since we're computing grads one by one, using top sort order.
            v.grad_fn.compute_and_add_gradient(v.grad)

    def relu(self):
        return relu(self)

    def tanh(self):
        return tanh(self)

    def abs(self):
        return abs_f(self)

    def __neg__(self):
        return self * -1

    def __add__(self, other: Union[Value, int, float]):
        other = Value(v=other) if not isinstance(other, Value) else other
        return add(self, other)

    def __mul__(self, other: Union[Value, int, float]):
        other = Value(v=other) if not isinstance(other, Value) else other
        return mul(self, other)

    def __truediv__(self, other: Union[Value, int, float]):
        other = Value(v=other) if not isinstance(other, Value) else other
        return div(self, other)

    def __radd__(self, other: Union[Value, int, float]):
        return self + other

    def __sub__(self, other: Union[Value, int, float]):
        return self + (-other)

    def __rsub__(self, other: Union[Value, int, float]):
        return (-self) + other

    def __rmul__(self, other: Union[Value, int, float]):
        return self * other

    def __rtruediv__(self, other: Union[Value, int, float]):
        other = Value(v=other) if not isinstance(other, Value) else other
        return other / self

    def __pow__(self, power: Union[Value, int, float], modulo=None):
        if modulo is not None:
            raise ValueError(f'Modulo operation is not supported')

        power = Value(v=power) if not isinstance(power, Value) else power
        return pow(self, power)

    def __repr__(self):
        return f"Value(v={self.v:.3f}, grad={self.grad:.3f}, grad_fn={type(self.grad_fn).__name__})"


def add(x1: Value, x2: Value) -> Value:
    res = x1.v + x2.v
    return Value(v=res, grad_fn=SumBackward(x1, x2))


def mul(x1: Value, x2: Value) -> Value:
    res = x1.v * x2.v
    return Value(v=res, grad_fn=MulBackward(x1, x2))


def div(x1: Value, x2: Value) -> Value:
    res = x1.v / x2.v
    return Value(v=res, grad_fn=DivBackward(x1, x2))


def pow(x1: Value, pow: Value) -> Value:
    if x1.v <= 0:
        raise ValueError(f'Non positive base {x1.v:.3f} is not supported during pow operation.')
    return Value(x1.v ** pow.v, grad_fn=PowBackward(x1, pow))


def relu(x1: Value) -> Value:
    return Value(v=max(0.0, x1.v), grad_fn=ReLUBackward(x1))


def tanh(x1: Value) -> Value:
    return Value(v=math.tanh(x1.v), grad_fn=TanhBackward(x1))


def abs_f(x1: Value):
    return Value(v=abs(x1.v), grad_fn=AbsBackward(x1))


def softmax(values: List[Value]) -> List[Value]:
    e = Value(math.e)
    exp_vals = [e ** v for v in values]
    sum_exp = sum(exp_vals)
    out = [v / sum_exp for v in exp_vals]
    return out


def transpose(input: List[List[Any]]) -> List[List[Any]]:
    """
    :return: same Value's objects but in the different shape.
    """
    return [list(patches) for patches in zip(*input)]


def reshape(input: List[Any], height: int, width: int) -> List[Any]:
    """
    :return: same Value's objects but in the different shape.
    """
    if len(input) != (size := width * height):
        raise ValueError(f'Expected size of the input is {size}, got {len(input)}')

    out = [input[i:i + width] for i in range(0, len(input), width)]
    return out


def flatten(arr: Iterable[Union[Any, Iterable]]) -> List[Any]:
    """
    :return: same Value's objects but in the different shape.
    """
    flattened = []
    for item in arr:
        if isinstance(item, Iterable):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened


def as_value_array(arr: Union[Iterable[Any]]):
    out = []
    for v in arr:
        if isinstance(v, Iterable):
            v = as_value_array(v)
            out.append(v)
        else:
            out.append(Value(float(v)))
    return out


def value_array_to_list(arr: Iterable[Any]):
    out = []
    for val in arr:
        if isinstance(val, Iterable):
            val = value_array_to_list(val)
            out.append(val)
        else:
            out.append(val.v)
    return out


class GradFunction:
    """
    Abstraction over function that can be differentiated, such as sum(x1, x2). The class instance should hold reference to the
    function's input for which we can compute gradient using compute_and_add_gradient method.
    """

    @abstractmethod
    def inputs(self) -> List[Value]:
        """
        :return: function inputs. E.g. if the GradFunction represents sum operation of the two elements y = x1 + x2, this method will return
                 the list [x1, x2].
        """
        pass

    @abstractmethod
    def compute_and_add_gradient(self, grad: float):
        """
        Computes and adds gradient to the function's input.

        :param grad: gradient of the function's output used during chain rule.
        """
        pass


class SumBackward(GradFunction):
    def __init__(self, x1: Value, x2: Value):
        self.x1 = x1
        self.x2 = x2

    def inputs(self) -> List[Value]:
        return [self.x1, self.x2]

    def compute_and_add_gradient(self, grad: float):
        self.x1.grad += grad
        self.x2.grad += grad


class MulBackward(GradFunction):
    def __init__(self, x1: Value, x2: Value):
        self.x1 = x1
        self.x2 = x2

    def inputs(self) -> List[Value]:
        return [self.x1, self.x2]

    def compute_and_add_gradient(self, grad: float):
        self.x1.grad += grad * self.x2.v
        self.x2.grad += grad * self.x1.v


class DivBackward(GradFunction):
    """
    Backward logic for x1 / x2
    """

    def __init__(self, x1: Value, x2: Value):
        self.x1 = x1
        self.x2 = x2

    def inputs(self) -> List[Value]:
        return [self.x1, self.x2]

    def compute_and_add_gradient(self, grad: float):
        self.x1.grad += grad * 1 / self.x2.v
        self.x2.grad += grad * -self.x1.v / (self.x2.v ** 2)


class PowBackward(GradFunction):
    """
    Backward logic for x1 ** pow
    """

    def __init__(self, x1: Value, pow: Value):
        self.x1 = x1
        self.pow = pow

    def inputs(self) -> List[Value]:
        return [self.x1, self.pow]

    def compute_and_add_gradient(self, grad: float):
        self.x1.grad += grad * self.pow.v * (self.x1.v ** (self.pow.v - 1))
        self.pow.grad += grad * (self.x1.v ** self.pow.v) * math.log(self.x1.v)


class ReLUBackward(GradFunction):
    def __init__(self, x1: Value):
        self.x1 = x1

    def inputs(self) -> List[Value]:
        return [self.x1]

    def compute_and_add_gradient(self, grad: float):
        if self.x1.v > 0:
            self.x1.grad += grad


class TanhBackward(GradFunction):
    def __init__(self, x1: Value):
        self.x1 = x1

    def inputs(self) -> List[Value]:
        return [self.x1]

    def compute_and_add_gradient(self, grad: float):
        self.x1.grad += grad * (1 - math.tanh(self.x1.v) ** 2)


class AbsBackward(GradFunction):
    def __init__(self, x1: Value):
        self.x1 = x1

    def inputs(self) -> List[Value]:
        return [self.x1]

    def compute_and_add_gradient(self, grad: float):
        if self.x1.v >= 0:
            self.x1.grad += grad
        else:
            self.x1.grad -= grad
