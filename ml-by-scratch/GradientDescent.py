'''
Gradient Descent is a technique used in machine learning to 
__optimize__ the problem of finding the best model for a 
given dataset.
'''
from sympy import limit, symbols

from typing import Callable


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
    return (f(x + h) - f(x))/h


def differenciate(f: Callable[[float], float],
                  x: float) -> float:
    
    h = symbols('h')
    expr = (f(x + h) - f(x))/h
    return limit(expr, h, 0)