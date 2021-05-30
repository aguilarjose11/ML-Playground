

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

def neuron_output(weights: Vector,
    inputs: Vector,
    activation: Callable[[float], float]):
    """Same as perceptron but with bias in weights"""

    return activation(dot(weights, inputs))

def feed_forward(net: Tensor,
    input: Vector,
    activation: Callable[[float], float]):
    """Solve neural layer"""

    outputs: Matrix = []

    for layer in net:
        input_with_bias = input + [1]
        output = [neuron_output(neuron, input_with_bias, activation)
        for neuron in layer]
        outputs.append(output)
        input = output

    return outputs


### Il mio versione

class Neuron:
    """A Single non-spiking neuron

    A model of a non-spiking simplified neuron which applies a linear
    function to given weights and inputs to then be passed through an
    activation function.
    These Neurons must have a defined weight list since the neuron 
    inputs are inferred from these values. Randomness can be created
    by the implementing NeuralLayer and NeuralNetwork classes.



    """

    w_ij: Vector
    activation: Callable

    def __init__(self, 
                 weights: Vector, 
                 activation: Callable[[float], float]):
        """Create new neuron

        Creates a new neuron using the given values.

        Parameters
        ----------
        weights: Vector
            Weight initializer list. Will be used by linear integration to 
            find final output activation.
        activation: Callable
            Activation function for neuron to apply after linear 
            integration.

        """

        # Commonly a linear function
        self.activation = activation
        # neural inputs insinuated from weights
        self.w_ij = weights


    def output(self, input: Vector) -> float:
        """Solve neuron with given activation function
        
        As part of traditional neural activation, neural integration 
        is used. This is represented by the following formula:
        ___    
        \\  I        x_i * w_i
        //_ i = 0      

        Where x_n is the vector of inputs and w_i is the vector of 
        neural weights for each presynaptic neuron i.
        """ 

        return self.activation(
               sum(x*w for x, w in zip(input, self.w_ij)))

class NeuralLayer:

    layer: List[Neuron]
    
    def __init__(self, 
                 size: int, 
                 inputs: int,
                 weights: Matrix=None,
                 activation: Callable[[float], float]=sigmoid,
                 bias: bool=True,
                 bias_weights: Vector=None):
        """Populates neural layers
        
        parameters
        ----------
        size: int
            Specifies number of neurons to create. If using bias, it
            excludes extra bias neuron.
        inputs: int
            Number of inputs per neuron. If using bias, an extra 
            weight is required. This can be passed by the 
            bias_weights vector. See bias_weights for random weights.
        weights: Matrix
            Weights for every neuron. len(weights) shall be the same
            as parameter size.
        activation: Callable=sigmoid
            Specifies activation function to use by each neuron.
        bias: bool=True
            Specifies whether bias is to be used.
        bias_weights: Vector=None
            Weights to be used to compute bias. If left empty when 
            bias parameter is True, random biases will be computed
            per the number of specified inputs. This parameter is 
            ignored if bias=False.
        """
        
        # Error checks
        assert size >= 1, f"Invalid Layer Shape {size}"
        assert len(weights) == size if weights else 1, \
f"""Number of weights given per neuron {len(weights)} 
not same as size of layer {size}"""
        # Shall ignore bias_weights if no biases
        assert len(bias_weights) == size if bias else True, \
f"Bias shape {len(bias_weights)} != layer shape {size}"

        self.bias = bias
        self.layer = []

        # Do we need randomized weights?
        neural_weights = weights
        if not neural_weights:
            neural_weights = [[random.random() for _ in range(inputs)] for _ in range(size)]

        # Extend neural weights to account for bias if given.
        if bias:
            if bias_weights:
                neural_weights = [w + [w_b] for w, w_b in zip(neural_weights, bias_weights)]
            else:
                neural_weights = [w + [random.random()] for w in neural_weights]
        # Populate the neural layer.
        for i in range(size):
            self.layer.append(Neuron(neural_weights[i], activation))


    def __len__(self) -> int:
        """Dunder method for length"""
        return len(self.layer)


    def __getitem__(self, 
                    key: int) -> Neuron:
        """Dunder method for indexing"""
        return self.layer[key]


    def __setitem__(self, 
                    key: int, 
                    value: Neuron) -> None:
        """Dunder method for seting with indexing"""
        self.layer[key] = value


    def solve(self,
              input: Vector) -> Vector:
        """Passes input through every neuron"""
        
        # Error checking
        if len(input) != len(self.layer[0].w_ij):
            raise(Exception(f"Invalid Input Shape: {len(input)} != {len(self.layer[0].w_ij)}"))
        
        # Net_i computation
        output = [n.output(input) for n in self.layer]
        # Accounts for output of no-input bias neuron.
        if self.bias:
            output + [1]
        
        return output


class PerceptronLayer(NeuralLayer):
    pass

# TODO Add  output layer function specification!
class NeuralNetwork:
    """

    To add a bias, add an extra weight.
    """

    network: List[NeuralLayer]

    def __init__(self, 
                 shape: Vector, 
                 weights: Tensor=None, 
                 activation: List[Callable[[float], float]]=sigmoid,
                 use_bias: bool=True,
                 bias_weights: Tensor=None):
        
        # Run checks:
        if len(shape) < 1:
            raise(Exception("Invalid Seed shape"))
        self.bias = use_bias
        # Create input layer. Input layer never uses bias
        self.network = [NeuralLayer(
            shape[0],
            1, 
            weights[0], 
            activation[0], 
            False)]

        if weights:
            for i, layer_size in enumerate(shape[1:]):
                self.network.append(NeuralLayer(
                    layer_size, 
                    shape[i], 
                    weights[i+1], 
                    activation[i+1], 
                    use_bias, 
                    bias_weights))
        else:
            # Create input weights
            for i, layer_size in enumerate(shape[1:]):
                self.network.append(NeuralLayer(
                    layer_size, 
                    shape[i], 
                    None, 
                    activation[i+1], 
                    use_bias, 
                    bias_weights))# TODO

    def solve(self, x: Vector) -> List[float]:
        # In case of usage of bias.
        tmp = x.copy()
        # Pass through input layer
        tmp = self.network[0].solve(tmp)
        # Pass through every other layer
        for layer in self.network[1:]:
            tmp = layer.solve(tmp)
        # Last element is bias if used. uneccesary.
        if self.bias:
            return tmp[:-1]
        else:
            return tmp