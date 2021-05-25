

from utils import Vector, Matrix, Tensor, dot, step_function, sigmoid
from typing import List, Callable
import random





### From-Scratch Book

def perceptron_output(weights: Vector, 
                      bias: float, 
                      x: Vector,
                      activation: Callable[[float], float] = step_function) -> float:
    """Perceptron networker"""
    net = dot(weights, x) + bias
    return activation(net)


### Il mio versione

class Neuron:
    """A Single non-spiking neuron
    A model of a non-spiking simplified neuron which applies a linear
    function to given weights and inputs to then be passed through an
    activation function.

    """

    w_ij: Vector
    activation: Callable

    def __init__(self, 
                 weights: Vector, 
                 activation: Callable[[float], float]):

        # Commonly a linear function
        self.activation = activation
        # pre-synaptic
        self.w_ij = weights


    def output(self, input: Vector) -> float:
        """Solve neuron with given activation function"""

        return self.activation(sum(x*w for x, w in zip(input, self.w_ij)))

class NeuralLayer:

    layer: List[Neuron]
    
    def __init__(self, 
                 size: float, 
                 inputs: float,
                 weights: List[Vector]=None,
                 activation: Callable[[float], float]=sigmoid):
        
        if size < 1:
            assert(Exception(f"Invalid Layer Shape {size}"))

        self.layer = []
        for i in range(size):
            weight = []
            if weights:
                weight.append(weights[i])## Working! not acceptins single elements!
            else:
                weight = [random.random() for _ in range(inputs)]
            
            self.layer.append(Neuron(weight, activation))

    def __len__(self) -> int:
        return len(layer)

    def __getitem__(self, key: int) -> Neuron:
        return self.layer[key]

    def __setitem__(self, key: int, value: Neuron) -> None:
        self.layer[key] = value

    def solve(self,
              input: Vector) -> Vector:
        """Passes input through every neuron"""
        return [n.output(input) for n in self.layer]

class PerceptronLayer(NeuralLayer):
    pass

class NeuralNetwork:
    """

    To add a bias, add an extra weight.
    """

    network: List[List[Neuron]]

    def __init__(self, 
                 shape: Vector, 
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