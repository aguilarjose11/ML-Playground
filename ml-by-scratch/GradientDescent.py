'''
Gradient Descent is a technique used in machine learning to 
__optimize__ the problem of finding the best model for a 
given dataset.
'''
from sympy import limit, symbols
import random

from typing import Callable, List
from linear_algebra import Vector, distance, add, scalar_multiply, dot


# Regular Differencial quotient used for differenciation:
'''
| *
| \*       Tangent line
|   \*
|     \ *
|____________*_____       _____________________
                         |       Quotient      |
(d/dx)*f(x) = lim_(h->0) [f(x + h) - f(x)] / h

'''
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    # this is the equation for a derrivative but without the limit.
    return (f(x + h) - f(x))/h


def differenciate(f: Callable[[float], float],
                  x: float) -> float:
    # full calculation of a derrivative with limit.
    h = symbols('h')
    expr = (f(x + h) - f(x))/h
    return limit(expr, h, 0)

# Partial derivatives as in multivariate calculus.
'''
let f(x, y, z, ...) map f: R x ... x R -> R. Assume f is differentiable; therefore, continuous:

df/dx = lim_(h->0) [f(x + h, y, z) - f(x, y, z)] / h 
df/dy = lim_(h->0) [f(x, y + h, z) - f(x, y, z)] / h 
df/dz = lim_(h->0) [f(x, y, z + h) - f(x, y, z)] / h 
...

Used to check the change in f while keeping some variable and changing one of the variable.

'''
def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,  # the ith differential
                                h: float) -> float:
    # We only add h to the specific ith partial derivative.
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v))/h

def partial_differenciate(f: Callable[[Vector], float],
                          v: Vector,
                          i: int) -> float:
    '''
    do the (i)th partial differential on v of f. 
    '''
    h = symbols('h')
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    expr = ((f(w) - f(v))/h)
    return limit(expr, h, 0)


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001) -> Vector:
    '''
        It is more of an art than science to pick the steps h. it is good to experiment with this.
    '''
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v: Vector,
                  gradient: Vector,
                  step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(step, v)

def minimum_approx(f: Callable[[Vector], float],
                   v: Vector,
                   h: int = 0.0001,
                   step_size: float = -0.01,
                   epoch: int = 1000) -> Vector:
    # note: the actual calculation may fail because the value given for h may not be small enough that the calculation will "hover" around result
    w = v.copy()
    for iteration in range(epoch):
        gradient = estimate_gradient(f, w, h)
        w = gradient_step(w, gradient, step_size)
        print(iteration, w)

def gradient_vector(f: Callable[[Vector], float],
                    v: Vector) -> Vector:
    '''
    let f(x, y, z, ...) map f: R x ... x R -> R. Assume f is differentiable; therefore, continuous:
    __
    \/ f(x, y, z, ...) = <df/dx, df/dy, df/dz, ...> 
    '''
    return [partial_differenciate(f, v, i) for i in range(len(v))]

def directional_differenciate(f: Callable[[Vector], float],
                           v: Vector,
                           u: Vector) -> float:
    ''' return the rate of change in the dirrection of the unit vector u
    let f(x, y, z, ...) map f: R x ... x R -> R. Assume f is differentiable; therefore, continuous:
    D_u f(x, y, z, ...) = \/f * u, where \/f is the gradient vector of f
    '''
    return dot(gradient_vector(f, v), u)

def minimum_approx_real(f: Callable[[Vector], float],
                        v: Vector,
                        h: int = 0.0001,
                        step_size: float = -0.01,
                        epoch: int = 1000) -> Vector:
    # uses actual gradient. Careful: computationally expensive!
    w = v.copy()
    for iteration in range(epoch):
        gradient = gradient_vector(f, w)
        w = gradient_step(w, gradient, step_size)
        print(iteration, w)

def minimum(f: Callable[[Vector], float],
            v: Vector,
            h: int = 0.0001,
            step_size: float = -0.01,
            epoch: int = 1000) -> Vector:
    # uses actual gradient. Careful: computationally expensive!



    w = v.copy()
    for iteration in range(epoch):
        gradient = gradient_vector(f, w)
        w = gradient_step(w, gradient, step_size)
        print(iteration, w)