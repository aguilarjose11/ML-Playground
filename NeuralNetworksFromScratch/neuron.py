

from utils import Vector, Matrix, Tensor
from typing import List, Callable


class Neuron:
    
    w_ij: Vector
    activation: Callable

    def __init__(self, weights: Vector, 
                 activation: Callable[[float], float]):
        self.activation = activation
        self.w_ij = weights


    def output(self, input: Vector) -> float:
        return sum(self.activation(x) for x in input)



class NeuralNetwork:
    """

    To add a bias, add an extra weight.
    """

    network: List[List[Neuron]]

    def __init__(self, 
                 seed: Vector, 
                 weights: Tensor, 
                 activation: List[Callable[[float], float]],
                 use_bias: bool=False):
        
        # Run checks:
        if len(seed) < 1:
            raise(Exception("Invalid Seed shape"))
        # Create input layer
        for i, l_seed in enumerate(seed):
            self.network[i].append([Neuron(w, activation[i]) for w in weights[i]])
            pass

    def solve(self, x: Vector):
        pass


class Perceptron:
    def solve