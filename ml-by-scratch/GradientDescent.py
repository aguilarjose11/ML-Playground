'''
Gradient Descent is a technique used in machine learning to 
__optimize__ the problem of finding the best model for a 
given dataset.
'''
from sympy import limit, symbols
import random

from typing import Callable, List, TypeVar, Iterator
from linear_algebra import Vector, distance, add, scalar_multiply, dot, vector_mean


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




def linear_gradient(x: float,
                    y: float,
                    theta: Vector) -> Vector:
    slope, intercept = theta # theta is changed to minimize the cost function. will contain the best parameters at the end of training.
    prediction = slope*x + intercept
    error = prediction - y
    squared_error = error ** 2

    '''
        The book is note very specific on how the gradient formula is calculated 
        and may not be as obvious at first for newbies. Here it is an explanation:

        error = y_p - y, where y_p is the prediction, and y is the actual value.

        This example uses the Squared Error so that we can calculate
        the Mean Absolute Error(MSE) error function:
        SE = error^2
        MSE = 1/n * sum(SE) , where n is the number of cases that were evaluated
        note: this builds for RMSE, which has the same unit as the y value. we won't see it here
        There are 2 reasons to why we do the gradient of the SE instead of the whole MSE:
        1. we want to find the *MEAN* gradient (its direction is that of greatest ascend, remember?).
           because of this, we need the individual SE calculations for all of the training data with
           which we will run the gradient descent algorithm to find the best model parameters (ax + b). It is just simpler.
        2. To complete the computation of MSE, we need SE. MSE will be the vector mean
           of the SE's computed from the training data.
        recall the formula for the gradient:
        \/ f(x, y, z, ...) = <df/dx, df/dy, df/dz, ...>

        in this case, f = SE(error), where error(y_p, y), where y_p = s*x + i (the model to be trained) where s = slope, i = intercept 
        therefore:
          error^2       (y_p - y)^2   (s*x + i - y)**2
        \/ SE(error) = \/SE(y_p, y) =     \/SE(s, i)    = <dSE/ds, dSE/di>
        dSE/ds = 2 * (s*x + i - y) * x = 2*(y_p - y)*x = 2*error*x
        dSE/di = 2 * (s*x + i - y) = 2*(y_p - y) = 2*error               |
        therefore:                                                       |
        \/SE(error) = <2*error*x, 2*error>, which in python is as below \|/
    '''
    se_gradient = [2 * error * x, 2 * error]
    return se_gradient


linear_model_data = [x for x in range(-50, 50)] 
linear_model_targets = [35*x + 15 for x in range(-50, 50)] 


def gradient_descent_linear_model(X: Vector,
                                  y: Vector,
                                  learning_rate: float = 0.001,
                                  iterations: int = 5000,
                                  show_case: bool = True) -> None:
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)] # Random start point.
    for epoch in range(iterations):
        # we gather the gradient ME error for every case. (which direction we can go to minimize error.)
        # remember that the values in the list are not SE error, but rather the dirrection
        # at which we should move to minimize wathever the error was. we are concerned with
        # this since we are TRAINING not TESTING!
        fit_se_err = [linear_gradient(data, actual, theta) for data, actual in zip(X, y)]
        # calculate the average dirrection where we can go to minimize the error MSE
        # here, this is not the calculated MSE, but just the average of all of
        # the gradients!
        mse_gradient = vector_mean(fit_se_err)
        # We calculate the step at which we want to move, following the gradient starting
        # at point theta and at the step size of learning rate in the direction of minimization
        # hence the negative.
        theta = gradient_step(theta, mse_gradient, -learning_rate)
        # this is just to show the progress in the training.
        if(show_case):
            print(f"{str(epoch)} - parameters: {str(theta)} -- mse average minimization gradient: {str(mse_gradient)}")
    # at this point, we will have the best parameters in theta for a linear model.


T = TypeVar('T')  # generic functions


def minibatch(dataset: List[T],
              batch_size: int,
              shuffle: bool = True) -> Iterator[List[T]]:
    """Generate batches of the specified size"""

    # creates a list of where each batch of batch_size size start at (we will be returning multiple!)
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle:
        random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size  # note: this code works because in python we don't get an error when we go out of bounds when slicing!
        yield dataset[start:end]


def minibatch_gradient_descent(X: List[T], 
                               y: List[T], 
                               batch_size: int=20,
                               learning_rate: float=0.001, 
                               iterations: float=1_000,
                               show_case: bool=True) -> None:
    """Mini batch gradient descent
    The application of this algorithm is rooted in the concept of gradient descent,
    but it deviates in that instead of calculating the gradient on the whole dataset,
    we instead split the dataset into pieces (batches) from which we calculate the
    gradient.

    the advantages to this approach is mainly visible when working with large datasets.
    while the computation seems to follow a O(n^2), the speed with which it moves towards
    the optimal training parameters is faster since steps towards it are more common than
    when having to compute the gradient for every single point in the dataset.
    Therefore, with this approach we can approach the optimal parameters for out models
    with less epochs than with regular gradient descent.

    """
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    for epoch in range(iterations):
            for batch in minibatch([(data, actual) for data, actual in zip(X, y)], batch_size):
                gradient_mu = vector_mean([linear_gradient(data, actual, theta) for data, actual in batch])
                theta = gradient_step(theta, gradient_mu, -learning_rate)
                if(show_case):
                    print(f"{str(epoch)} - parameters: {str(theta)} -- mse average minimization gradient: {str(gradient_mu)}")

def stochastic_gradient_descent(X: List[T], 
                                y: List[T], 
                                batch_size: int=20,
                                learning_rate: float=0.001, 
                                iterations: float=1_000,
                                show_case: bool=True) -> None:
    """ Stochastic gradient descent (SGD)
    This technique of training is very similar to that of minibatch gradient descent.
    Many of its benefits are shared, but again, the time it takes to approach the optimal
    parameters for a model are smaller due to taking steps for every training example.
    A large trade off is that while we can find the optimal parameters much faster, the
    algorithm is much more suceptible to outliers (these may have a gradient away from
    what the whole dataset as a whole follows.)
    """
    # starting random point
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    for epoch in range(iterations):
        for data, actual in zip(X, y):
            gradient = linear_gradient(data, actual, theta)
            theta = gradient_step(theta, gradient, -learning_rate)
            if(show_case):
                    print(f"{str(epoch)} - parameters: {str(theta)} -- mse average minimization gradient: {str(gradient)}")

# https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants
# good comparison between all three.