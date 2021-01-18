'''
The use of Principal Component Analysis (PCA) is most exclusively
used for the problem of dimensionality reduction in Machine learning.

The math behind PCA

PCA aims at finding the dimensions (or components) that have the most
variance to then extract them.

1.- We normalize/de-mean the data
    * by finding the mean of the whole dataset, every point gets this
      value subtracted, so to bring the mean to 0. This is very
      important since PCA can (and will) identify the mean and create
      incorrect results.
2.- Use dirrectional Variance.
    * We build on top of statistic's variance formula:
    __
    \
    /_i  (x_i - x^bar)^2
    -------------------   =
           n - 1
                                                 _
    Because we have de-meaned the data, x^bar or x = 0
    __
    \
    /_i  (x_i)^2
    -------------------   =
           n - 1
    In order to find the dirrectional variance, we use the dot product,
    which happily lets us project the variance into some direction w_dir
    Note: * is the dot product here.
    __
    \
    /_i  (x_i * w_dir)^2
    --------------------   =  Direcctional Variance formula
           n - 1

    Notice how we do use bessel's correction (n-1). This comes from
    two schools of thought in mathematics: Statistics and Machine learing
    Statistics: account for the bias in the models created (see bassel)
    ML: We are trying to find the best model using usually Squared error.

    From my personal research in the topic, it seems rather ok to use both.
    Me, coming from a rather mathematical background I will use it in my
    mathematics. If you have arguments agains this, let me know. I always
    welcome anyone with suggestions.

    Finally, using the above mathematical description, we can find the 
    principal components by using gradient descent and finding the maxima.

    We ought to find the partial differential equation for the dirrectional
    variance if we want to find the gradient, for the gradient give us the
    direction of steppest descent/ascent. We only use the summation and
    discard the bottom of the fraction.
    \/f = 2*(x_i * w_dir)*x_i
'''


from linear_algebra import magnitude, dot, subtract, Vector, vector_mean, scalar_multiply
from GradientDescent import gradient_step

from typing import List

import tqdm


def de_mean(data: List[Vector]) -> List[Vector]:
    """Recenter data to 0"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]


def direction(w: Vector) -> Vector:
    """ Find direction vector"""
    mag = magnitude(w)
    return [w_i / mag for w_i in w]

def directional_variance(data: List[Vector], w: Vector) -> float:
    """Find the directional variance. check code description for math"""
    w_dir = direction(w)
    return sum(dot(v, w_dir)**2 for v in data)


def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data) for i in range(len(w))]


def first_principal_component(data: List[Vector],
                              n: int = 100,
                              step_size: float = 0.1):
    # Stochastic-like
    guess = [1.0 for _ in data[0]]
    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient,
                                  t.set_description(f"dv: {dv:.3f}"))
    return direction(guess)


def project(v: Vector, w: Vector) -> Vector:
    """return Proj_w(v)"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length)


def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """removes v from its projection onto w"""
    return subtract(v, project(v, w))


def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    """Removes the projection of data onto w"""
    return [remove_projection_from_vector(v, w) for v in data]


def pca(data: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)
    return components


def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]


def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]
