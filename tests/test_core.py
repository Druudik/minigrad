import math
import unittest
from typing import Tuple, List, Union, Callable

import numpy as np
import torch

from minigrad.core import Value, reshape, transpose, softmax

T = Union[Value, torch.Tensor]


class TestCore(unittest.TestCase):

    # noinspection PyUnresolvedReferences
    def test_basic_operations(self):
        a = Value(v=1)
        b = a + 1
        self.assertEqual(1, a.v)
        self.assertEqual(2, b.v)

        c = a * b - 1 + 2 * a
        self.assertEqual(3, c.v)
        self.assertEqual(3, c.relu().v)

        d = -c
        self.assertEqual(-3, d.v)
        self.assertEqual(0, d.relu().v)

        e = (3 - d - 1) * b
        self.assertEqual(10, e.v)

        e = e ** 2.5
        self.assertAlmostEqual(316.227766017, e.v, places=5)

    def test_complex_operations_and_gradients(self):
        """
        Tests more complex operations on the Value class, including gradient computation. PyTorch is used as a reference
        for the correctness.
        """

        # noinspection PyTypeChecker
        def _complex_logic(v: T, softmax: Callable) -> Tuple[T, List[T]]:
            """
            :return: final output and all intermediate values that were created during computation.
            """
            a = v * 2
            b = a + 2 * (-v) + 1
            c = 3 / b
            d = 1 + b + c / (a + 1)
            e = (d.relu() + a.relu()).relu() * 5 - d - c * 0.2
            f = (e - 2) / 1.5 - c + a
            g = (f.relu() + 1) ** 2.1
            h = (g.relu() + 1) ** -5
            i = (-g).abs() + g.abs() + h.tanh() * 0.3
            j = -(v + v * v) / (v + 1000)

            softmax_out = softmax([c / 1000, d / 1000, e / 500])
            all_values = [a, b, c, d, e, f, g, h, i, j, *softmax_out]
            out: T = sum(all_values) / len(all_values)
            return out, all_values

        values_to_test = np.array([0, 0.1, 0.53, 0.78, 1, math.pi, 3.75, 5, 10, 100, 1001], dtype=np.float64)
        values_to_test = np.concatenate([values_to_test, -values_to_test, values_to_test * 5.75468])

        for v in values_to_test:
            with self.subTest(v=v):
                # Compute outputs and intermediate results for both torch.Tensor and Value.
                output, intermediate = _complex_logic(Value(v), softmax)

                torch_output, torch_intermediate = _complex_logic(
                    v=torch.tensor([v], dtype=torch.float64, requires_grad=True),
                    softmax=lambda vals: torch.softmax(torch.tensor(vals, requires_grad=True), dim=0)
                )

                # We need to call retain_grad on the torch tensors, so that we're able to
                # access their grad attribute later.
                torch_output.retain_grad()
                for t in torch_intermediate:
                    t.retain_grad()

                # Compare outputs and intermediate results.
                self.almost_equal(output, torch_output)
                self.all_close(intermediate, torch_intermediate)

                # Compute and compare gradients.
                torch_output.backward()
                output.backward()
                self.almost_equal(output, torch_output)
                self.all_close(intermediate, torch_intermediate)

    # noinspection PyTypeChecker
    def test_reshape(self):
        arr = np.arange(12)
        out = reshape(arr, 4, 3)
        np.testing.assert_array_equal(arr.reshape(4, 3), out)

    # noinspection PyTypeChecker
    def test_transpose(self):
        matrix = np.arange(12).reshape(3, 4)
        out = transpose(matrix)
        np.testing.assert_array_equal(matrix.T, out)

    def almost_equal(self, actual: Value, expected: torch.Tensor, decimal: int = 9):
        """
        Checks the gradient and value equality.
        """
        np.testing.assert_almost_equal(
            actual.v,
            expected.item(),
            decimal=decimal,
        )
        np.testing.assert_almost_equal(
            actual.grad,
            expected.grad.item() if expected.grad is not None else 0,
            decimal=decimal,
        )

    def all_close(self, actual: List[Value], expected: List[torch.Tensor], atol: float = 1e-9):
        """
        Checks the gradient and value equality for every pair.
        """
        np.testing.assert_allclose(
            [v.v for v in actual],
            [t.item() for t in expected],
            atol=atol,
        )
        np.testing.assert_allclose(
            [v.grad for v in actual],
            [(t.grad.item() if t.grad is not None else 0) for t in expected],
            atol=atol,
        )


if __name__ == '__main__':
    unittest.main()
