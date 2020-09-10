'''
Linear Algebra used for machine learning.
'''
import math
from typing import List, Tuple, Callable

#----------------
# Vector Algebra |
#----------------

# Vectors store a single row of a matrix.
Vector = List[float]


# Vectorial Addition.
def add(v: Vector, w: Vector) -> Vector:
    '''
    with:
        v = (v_1, v_2, ..., v_x)
        w = (w_1, w_2, ..., w_x)

    v + w = (v_1 + w_1, v_2 + w_2, ..., v_x + w_x)
    as:
        add(v, w)
    '''
    assert len(v) == len(w), f"Dimension of vectors do not match: dim(v)={len(v)} != dim(w)={len(w)}."
    return [v_x + w_x for v_x, w_x in zip(v, w)]


# Vectorial Subtraction.
def subtract(v: Vector, w: Vector) -> Vector:
    '''
    with:
        v = (v_1, v_2, ..., v_x)
        w = (w_1, w_2, ..., w_x)

    v - w = (v_1 - w_1, v_2 - w_2, ..., v_x - w_x)
    as:
        subtract(v, w)
    '''
    assert len(v) == len(w), f"Dimension of vectors do not match: |v|={len(v)} != |w|={len(w)}."
    return [v_x - w_x for v_x, w_x in zip(v, w)]


# Multiple vector addition.
def vector_sum(vectors: List[Vector]) -> Vector:
    '''
    for every vector v_x having same dimensions:
        v_1 + v_2 + ... + v_x = 
        (v_1x + v_2x + ... v_xx, v_1y + v_2y + ... v_xy, v_1z + v_2z + ... v_xz)
    '''
    assert vectors, "No vectors provided!"
    n_vector_dim = len(vectors[0])
    assert all(len(v) == n_vector_dim for v in vectors), "dimensions do not match!"

    return [sum(vector[x] for vector in vectors)
            for x in range(n_vector_dim)]


# Scalar Product.
def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_n for v_n in v]


# Component-Wise vectorial mean.
def vector_mean(vectors: List[Vector]) -> Vector:
    assert vectors, "No vectors provided!"
    n_vector_dim = len(vectors[0])
    assert all(len(v) == n_vector_dim for v in vectors), "dimensions do not match!"
    return scalar_multiply(1/len(vectors), vector_sum(vectors))


# dot product.
def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "vector sizes are different!"
    return sum(v_n * w_n for v_n, w_n in zip(v, w))


# Summation of squares.
def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


# Vectorial Magnitude.
def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))


# Vectorial distance.
def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))


# ----------------
# Matrix Algebra |
# ----------------

# Note: Matrix = List[Vector]
Matrix = List[List[float]]


# Matrix shape
def shape(A: Matrix) -> Tuple[int, int]:
    return len(A), len(A[0])


# Matrix row extraction
def get_row(A: Matrix, i: int) -> Vector:
    return A[i]


# Matrix column extraction
def get_column(A: Matrix, j: int) -> Vector:
    return [column[j] for column in A]


# Matrix generation with specified dimensions
def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(row, col) for col in range(num_cols)] for row in range(num_rows)]


# Identity Matrix generator.
def identity_matrix(n: int) -> Matrix:
    def identity_matrix_base_func(row, col):
        if row == col:
            return 1
        else:
            return 0
    return make_matrix(n, n, identity_matrix_base_func)


# Matrix scalar product
def matrix_scalar_multiply(c: float, A: Matrix) -> Matrix:
    return [scalar_multiply(c, row) for row in A]


def transpose(A: Matrix) -> Matrix:
    return [[row[col] for row in A] for col in range(len(A[0]))]


# Matrix cross-product
def matrix_dot(A: Matrix, B: Matrix) -> Matrix:
    assert shape(A)[1] == shape(B)[0], "Inner dimensions of matrices do not match!"

    return [[dot(row, column) for column in transpose(B)] for row in A]