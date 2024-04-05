import unittest

import numpy as np
import torch

from minigrad.core import relu, tanh, Value, as_value_array, flatten, value_array_to_list
from minigrad.nn import Neuron, Conv2D


class TestNN(unittest.TestCase):

    def test_neuron(self):
        n = Neuron(3, activation=relu)

        # Custom weight initialization that's needed for the testing w=[1, 2, 3] and bias=1.
        for i, weight in enumerate(n.weight):
            weight.v = i + 1
        n.bias = 1

        input = as_value_array([1, 3, 4.1])
        out = n.forward(input)
        np.testing.assert_almost_equal(out.v, 20.3, decimal=5)

        input = as_value_array([1, 3, -4.2])
        out = n.forward(input)
        np.testing.assert_equal(out.v, 0)

    # noinspection PyTypeChecker
    def test_conv2d(self):
        """
        Test Conv2D class with 3 output channels, kernel size 2, stride 2 and input image dimensions [2, 4, 4].
        The output values are compared to the PyTorch Conv2D class that was initialized with the same weights.
        """
        input = np.array([
            [1.0, 2.3, 4.7, 5.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 2.3, 1.7, 1.0],
        ])
        input = np.stack([input, input - 2])

        # Create filters with the similar weights for each input channel.
        init_filters_weight = np.array([
            [[[1.0, 0.2],
              [0.1, 1.0]]] * 2,

            [[[3.0, 1.2],
              [1.0, 0.0]]] * 2,

            [[[0.0, 0.7],
              [0.7, 3.2]]] * 2,
        ])
        init_filters_weight[:, 1, :, :] -= 0.01
        init_bias = np.array([0.1, 0.5, 1.0])

        # Initialize Conv2D weights.
        conv = Conv2D(4, 4, kernel_size=2, stride=2, in_channels=2, out_channels=3, activation=tanh)
        for i, n in enumerate(conv.filters.neurons):
            self.assertEqual(8, len(n.weight))
            n.weight = as_value_array(flatten(init_filters_weight[i]))
            n.bias = Value(init_bias[i])

        # Compute output.
        out = conv.forward(as_value_array(input))

        # Compute torch reference output. torch.conv2d expects the batched input, so we need to add new dim at the beginning.
        torch_out = torch.conv2d(
            torch.from_numpy(input[np.newaxis, ...]),
            torch.from_numpy(init_filters_weight),
            bias=torch.from_numpy(init_bias),
            stride=2,
        )[0]
        torch_out = torch_out.tanh()

        # Compare output to the torch tensors.
        out = value_array_to_list(out)
        np.testing.assert_allclose(out, torch_out, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
