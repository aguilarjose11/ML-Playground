import pytest
from linear_algebra import *
import math


def test_vectorial_addition() -> None:

    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    c = [-1.0, -5.0, -10.0]
    d = [1.0, 5.0, 10.0]
    e = [5.0]

    assert add(a, b) == [6, 8, 10]
    assert add(c, d) == [0, 0, 0]
    with pytest.raises(Exception):
        add(d, e)
    
def test_vectorial_subtraction() -> None:

    a = [1.0, 2., 3.]
    b = [5., 6., 7.]
    c = [-1., -5., -10.]
    e = [5.]

    assert subtract(a, b) == [-4, -4, -4]
    assert subtract(c, c) == [0, 0, 0]
    with pytest.raises(Exception):
        subtract(c, e)
    
def test_multiple_vectorial_sum() -> None:
    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    c = [-1.0, -5.0, -10.0]
    d = [1.0, 5.0, 10.0]
    e = [5.0]
    x = [a, b, c, d]

    assert vector_sum(x) == [6., 8., 10.]
    assert vector_sum([a, b]) == add(a, b)
    with pytest.raises(Exception):
        vector_sum([])
    with pytest.raises(Exception):
        vector_sum([a, b, e])
    
def test_scalar_product() -> None:
    a = [1.0, 2.0, 3.0]
    b = -6
    c = 4

    assert scalar_multiply(b, a) == [-6, -12, -18]
    assert scalar_multiply(c, a) == [4, 8, 12]


def test_component_wise_vectorial_mean() -> None:
    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    c = [-1.0, -5.0, -10.0]
    d = [1.0, 5.0, 10.0]

    assert vector_mean([a, b, c, d]) == [6/4, 8/4, 10/4]

def test_dot_product():
    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    e = [5.0]

    assert dot(a, b) == 38.0
    with pytest.raises(Exception):
        dot(a, e)

def test_sum_of_squares():

    a = [1.0, 2.0, 3.0]
    result = 1**2 + 2**2 + 3**2

    assert sum_of_squares(a) == result

def test_magnitude():
    a = [2, 3, 4]
    result = math.sqrt(2**2 + 3**2 + 4**2)
    assert magnitude(a) == result

def test_distance():
    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    result = math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

    assert distance(a, b) == result
    
def test_matrix_shape():
    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    c = [-1.0, -5.0, -10.0]
    d = [1.0, 5.0, 10.0]
    A = [a, b, c, d]
    result = 4, 3

    assert shape(A) == result

def test_get_row():
    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    c = [-1.0, -5.0, -10.0]
    d = [1.0, 5.0, 10.0]
    A = [a, b, c, d]
    result = c

    assert get_row(A, 2) == result

def test_get_column():
    a = [1.0, 2.0, 3.0]
    b = [5.0, 6.0, 7.0]
    c = [-1.0, -5.0, -10.0]
    d = [1.0, 5.0, 10.0]
    A = [a, b, c, d]
    result = [3, 7, -10, 10]

    assert get_column(A, 2) == result

def test_matrix_generator_from_function():
    entry_fn_0 = lambda row, col: 0
    entry_fn_1 = lambda row, col: 1 if row == col else 0
    result_0 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    result_1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    assert make_matrix(3, 4, entry_fn_0) == result_0
    assert make_matrix(4, 4, entry_fn_1) == result_1

def test_matrix_generator_from_function():
    entry_fn = lambda row, col: 1 if row == col else 0
    result_0 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    result_1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    assert make_matrix(3, 3, entry_fn) == result_0
    assert make_matrix(4, 4, entry_fn) == result_1

def test_matrix_scalar_product():
    A = [
        [1,2,3],
        [4,5,6],
        [7,8,9],
    ]
    c = 3
    result = [
        [3, 6, 9],
        [12, 15, 18],
        [21, 24, 27],
    ]

    assert matrix_scalar_multiply(c, A) == result

def test_matrix_transpose():
    #2x4
    A =[
        [3, 4, 1, 9],
        [4, 2, 4, 10],
    ]
    #3x3
    B = [
        [9, 7 ,4],
        [1, 3, 2],
        [10, 21, 11],
    ]
    result_A =[
        [3, 4],
        [4, 2],
        [1, 4],
        [9, 10],
    ]
    result_B = [
        [9, 1, 10],
        [7, 3, 21],
        [4, 2, 11],
    ]

def test_matrix_dot_product():
    #3x3
    A = [
        [2, 3, 7],
        [1, 2, 0],
        [1, 2, 2],
    ]
    #3x2
    B = [
        [3, 4],
        [6, 1],
        [9, 10],
    ]
    #2x3
    C = [
        [2, 4, 9],
        [5, 7, 21],
    ]

    BC = [
        [3*2 + 4*5, 3*4 + 4*7, 3*9 + 4*21],
        [6*2 + 1*5, 6*4 + 1*7, 6*9 + 1*21],
        [9*2 + 10*5, 9*4 + 10*7, 9*9 + 10*21],
    ]
    CB = [
        [2*3 + 4*6 + 9*9, 2*4 + 4*1 + 9*10],
        [5*3 + 7*6 + 21*9, 5*4 + 7*1 + 21*10],
    ]
    AB = [
        [2*3 + 3*6 + 7*9, 2*4 + 3*1 + 7*10],
        [1*3 + 2*6 + 0*9, 1*4 + 2*1 + 0*10],
        [1*3 + 2*6 + 2*9, 1*4 + 2*1 + 2*10],
    ]


    with pytest.raises(Exception):
        matrix_dot(A, C)

    assert shape(matrix_dot(B, C)) == (3, 3)
    assert shape(matrix_dot(C, B)) == (2, 2)
    assert shape(matrix_dot(A, B)) == (3, 2)
    assert matrix_dot(B, C) == BC
    assert matrix_dot(C, B) == CB
    assert matrix_dot(A, B) == AB