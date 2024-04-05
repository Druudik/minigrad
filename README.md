# minigrad

Minimalistic pure Python implementation of the scalar-valued autograd engine with PyTorch-like API and various neural network layers.

Inspired by the Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Installation

Download the source code and run pip install from the root directory:
```bash
pip install .
```

One can also pip install directly from GitHub:
```bash
pip install git+https://github.com/Druudik/minigrad 
```

Requires Python >= 3.9

## Example usage

Using `Value` abstraction over scalar:
```python
from minigrad import Value

a = Value(1.0)
b = a * 2
c = (b - 3) / (a + 1.5).relu() + 5
d = a - (b - 7.1).abs() + c ** 2
d.backward()

# a.grad contains value of the ∂a/∂d (the same holds analogically for the b.grad, c.grad and d.grad). 
print(a.grad, b.grad, c.grad, d.grad)
```

Using neural network layers from `minigrad.nn` module:
```python
import numpy as np
import minigrad.nn as nn
from minigrad.core import as_value_array

img = as_value_array(np.random.rand(1, 15, 15))
conv = nn.Conv2D(width=15, height=15, kernel_size=5, stride=2, in_channels=1, out_channels=3)
out = conv.forward(img)
```

## Implementation

Whole autograd logic is implemented in one module [core.py](./minigrad/core.py)

At its core is `Value` class which wraps int/float. It provides support for various functions like addition, division or tanh.

Each `Value` is treated as a constant, so for every function's output, a new `Value` object is created. To allow dynamic tracking of 
the computational graph, its `grad_fn` field is populated by a reference to the derivative of the 
function and to the reference of its inputs (this is wrapped under one class `GradFunction`). 

All of this allows for a simple implementation of the backpropagation, which can be triggered by the `backward` method.

## Neural network training

The jupyter notebook [mnist_sgd.ipynb](mnist_sgd.ipynb) contains **end-to-end example** of the `minigrad` application to the MNIST dataset. 
The trained model consists of the two Conv2D layers and an MLP head with the tanh activations (goal was to use similar network as in
[LeCun 1989](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)). The expected input is 15x15 image with one channel, whose 
values are normalized to the zero mean and unit variance. 

The json checkpoint of the trained model weights can be found in the `artifacts` folder. It achieved 91.95% accurracy on the test set,
after 2 epochs, using SGD with batch size 2 (note that the training was stopped early, even though it still did not converge,
because the goal was to show just the simple end-to-end example and the `minigrad` implementation is kind of slow). 

To run the notebook, install its dependencies via:
```bash
pip install -r requirements.txt
```

## Running tests
Implementation of the `minigrad` is fully tested against PyTorch, which serves as a reference for the correct computation.

To run the tests, install PyTorch and NumPy and use:
```bash
python -m unittest discover ./tests
```

## Remarks
Since the goal of this project was very simple autograd engine in pure Python, the performance requirements went really sideways,
which resulted in the rather slow computation speeds. Just a little "be patient" warning 😄

NN layers support predictions for only 1 input (batching is achieved by computing prediction for each sample independently
and averaging their loss).

## License
[MIT](LICENSE)
