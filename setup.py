from setuptools import setup, find_packages

setup(
    name='minigrad',
    author='Kristián Kuľka',
    description='Minimalistic pure Python implementation of the scalar-valued autograd engine with PyTorch-like API and '
                'various neural network layers built on top of it.',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.9',
)
