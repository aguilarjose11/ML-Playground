""" Provides mathematical and other tools.

The utils.py package.
=====================
This package contains essential classes, functions, and variables
vital for the development of neural networks.

global variables
----------------


data types
----------

Vector -> list
- Represents mathematical vector.


functions
----------

dot -> float
* Performs mathematical dot product.
* param
    - x -> Vector
    - y -> Vector


classes
-------


"""

from typing import List
import math


# Datatypes
"""Mathematical Vector concept

The Vector can be seen as a list of numbers. If seen as a matrix, the
vector will be a matrix of shape 1 x N where N is the number of 
elements. Treat as a list.
"""
Vector = List[float]

Matrix = List[List[float]]

Tensor = List[List[List[float]]]


# Functions
def dot(x: Vector, y: Vector) -> float:
    """Mathematical dot product
    Performs the summation of the products of each element in each 
    matrix:
    ___ 
    \\  N      ( x_i * y_i )
    //_ i = 0

    """
    return sum(a*b for a, b in zip(x, y))


def step_function(x: float):
    """ Mathematical step function
    A type of indicatio function/characteristic function.
    """
    return 1.0 if x >= 0 else 0.0


def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))


def h_tan(t: float) -> float:
    return math.tanh(t)

def identity(t: float) -> float:
    return t