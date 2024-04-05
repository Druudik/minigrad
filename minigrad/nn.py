"""
Implementation of the layers such as MLP, Conv2D etc.
"""
import math
import random
from abc import abstractmethod
from typing import Any, List, Callable, Iterator, Optional

from minigrad.core import Value, ValueMatrix, flatten, transpose, reshape


class Module:

    @abstractmethod
    def get_parameters(self) -> List[Value]:
        """
        :return: flattened list of all the module's parameters (including those of submodules).
        """
        pass

    @abstractmethod
    def forward(self, *args: Any) -> Any:
        pass


class Neuron(Module):

    def __init__(self, in_dim: int, activation: Optional[Callable[[Value], Value]] = None):
        self.in_dim = in_dim

        # TODO this can be improved by using e.g. kaiming uniform init weight with custom gain per activation function type.
        #  See https://pytorch.org/docs/stable/nn.init.html for more details.

        # Taken from https://github.com/karpathy/lecun1989-repro, assuming tanh activation.
        sqrt_in_dim = math.sqrt(in_dim)
        weight_range = 2.89 / sqrt_in_dim
        self.weight = [Value(v=random.uniform(-weight_range, weight_range)) for _ in range(in_dim)]
        self.bias = Value(v=0)

        self.activation = activation

    def get_parameters(self) -> List[Value]:
        return [*self.weight, self.bias]

    def forward(self, input: List[Value]) -> Any:
        if len(input) != self.in_dim:
            raise ValueError(f'Expected size of the input dimension is {self.in_dim}, got {len(input)}')

        out = sum((w_i * x_i for w_i, x_i in zip(self.weight, input)), self.bias)
        out = self.activation(out) if self.activation is not None else out
        return out


class Linear(Module):

    def __init__(self, in_dim: int, out_dim: int, activation: Optional[Callable[[Value], Value]] = None):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.neurons = [Neuron(in_dim, activation) for _ in range(out_dim)]

    def get_parameters(self) -> List[Value]:
        return flatten(n.get_parameters() for n in self.neurons)

    def forward(self, input: List[Value]) -> List[Value]:
        if len(input) != self.in_dim:
            raise ValueError(f'Expected size of the input dimension is {self.in_dim}, got {len(input)}')

        out = [n.forward(input) for n in self.neurons]
        return out


class Conv2D(Module):
    def __init__(
        self,
        width: int,
        height: int,
        kernel_size: int,
        stride: int,
        in_channels: int,
        out_channels: int,
        activation: Optional[Callable[[Value], Value]] = None
    ):
        # For the simplicity, we do not support padding which means that width and height must be higher than the kernel_size
        # and that every patch must fit into the image.
        if stride <= 0:
            raise ValueError(f'Invalid stride {stride}')

        if height < kernel_size or width < kernel_size:
            raise ValueError(f'Kernel size must be less than image dimensions.')

        if (height - kernel_size) % stride != 0 or (width - kernel_size) % stride != 0:
            raise ValueError(f'Combination of kernel and stride must precisely fit the image. No padding is supported.')

        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.out_width = (width - kernel_size) // stride + 1
        self.out_height = (height - kernel_size) // stride + 1
        self.output_size = self.out_width * self.out_height * self.out_channels

        self.filters = Linear(in_channels * kernel_size * kernel_size, out_channels, activation)

    def get_parameters(self) -> List[Value]:
        return self.filters.get_parameters()

    def forward(self, input: List[ValueMatrix]) -> List[ValueMatrix]:
        """
        :param input: shape [in_channels, height, width]
        :return: shape [out_channels, out_height, out_width]
        """
        if len(input) != self.in_channels:
            raise ValueError(f'Expected size of the input channels is {self.in_channels}, got {len(input)}')

        out = []
        # The patches are traversed horizontally starting from the top-left corner (0, 0) and then vertically.
        for patch in self._get_flattened_patches(input):
            # We apply all filters at once using linear layer which produces one Value for each filter.
            filters_output = self.filters.forward(patch)
            assert len(filters_output) == self.out_channels
            out.append(filters_output)

        # 'out': [patches, out_channels], 'reshaped_out': [out_channels, patches].
        reshaped_out = transpose(out)
        assert len(reshaped_out) == self.out_channels and len(reshaped_out[0]) == len(out)

        # 'reshaped_out': [out_channels, out_height, out_width]
        reshaped_out = [
            # Convert flattened 1D 'channel' (feature map) to the 2D matrix of shape [out_height, out_width].
            reshape(channel, self.out_height, self.out_width)
            for channel in reshaped_out
        ]

        return reshaped_out

    def _get_flattened_patches(self, input: List[ValueMatrix]) -> Iterator[List[Value]]:
        """
        :param input: shape is [in_channels, height, width].
        :return: iterator of the strided input's patches with shape [in_channels, kernel_size, kernel_size]
        """
        for h in range(0, self.out_height * self.stride, self.stride):
            for w in range(0, self.out_width * self.stride, self.stride):
                yield flatten(
                    row[w:w + self.kernel_size]
                    for channel in input
                    for row in channel[h:h + self.kernel_size]
                )
