from .neuron import Neuron, NeuralLayer, NeuralNetwork, feed_forward, perceptron_output
from .utils import Vector, Matrix, Tensor, dot, sigmoid
import random
import pytest

class TestNeuron:
    def test_init_nrand(self):
        """neuron initialization not random"""
        
        weights = [2, 3, 4, 3, 2]
        activation = sigmoid
        neuron = Neuron(weights=weights,
            activation=activation)

        assert neuron.w_ij == weights
        assert neuron.activation == activation


    def test_output(self):
        """neural activation"""
        
        presyn = 5
        weights = [random.random() for _ in range(presyn)]
        activation = sigmoid
        inputs = [random.random() for _ in range(presyn)]
        neuron = Neuron(weights=weights,
            activation=activation)
        true_val = perceptron_output(weights=weights,
            bias=0,
            x=inputs,
            activation=activation)
        result = neuron.output(input=inputs)

        assert result == result

    
    def test_len(self):

        presyn = 5
        weights = [random.random() for _ in range(presyn)]
        activation = sigmoid
        neuron = Neuron(weights=weights,
            activation=activation)
        assert len(neuron) == len(neuron.w_ij)


    def test_getitem(self):

        presyn = 5
        weights = [random.random() for _ in range(presyn)]
        activation = sigmoid
        neuron = Neuron(weights=weights,
            activation=activation)

        rand_index = random.randint(0, len(neuron.w_ij)-1)

        assert neuron[rand_index] == neuron.w_ij[rand_index]


    def test_setitem(self):

        presyn = 5
        weights = [random.random() for _ in range(presyn)]
        activation = sigmoid
        neuron = Neuron(weights=weights,
            activation=activation)

        rand_index = random.randint(0, len(neuron.w_ij)-1)

        w = random.randint(0, len(neuron.w_ij)-1)

        neuron[rand_index] = w

        assert neuron.w_ij[rand_index] == w
        


class TestNeuralLayer:
    def test_init(self):

        size = random.randint(2, 30)
        inputs = random.randint(1, 30)
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid
        # No bias
        bias = False
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)

        assert nl.bias == bias
        assert len(nl.layer) == size
        rand_index = random.randint(0, size-1)
        assert nl.layer[rand_index].w_ij == weights[rand_index]
        assert nl.layer[rand_index].activation == activation

        # With bias
        bias = True
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)

        assert nl.bias == bias
        assert len(nl.layer) == size
        rand_index = random.randint(0, size-1)
        assert nl.layer[rand_index].w_ij == weights[rand_index] + [bias_weights[rand_index]]
        assert nl.layer[rand_index].activation == activation

        # Single neuron layer no bias
        size = 1
        inputs = random.randint(1, 30)
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid
        # No bias
        bias = False
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)
        
        assert len(nl.layer) == 1
        rand_index = random.randint(0, size-1)
        assert nl.layer[rand_index].w_ij == weights[rand_index]
        assert nl.layer[rand_index].activation == activation


        # With bias
        bias = True
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)
        
        assert len(nl.layer) == 1 # Bias neuron not created by design
        rand_index = random.randint(0, size-1)
        assert nl.layer[rand_index].w_ij == weights[rand_index] + [bias_weights[rand_index]]
        assert nl.layer[rand_index].activation == activation

        
        

    def test_init_err(self):

        size = -1
        inputs = random.randint(1, 30)
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid

        bias = False
        bias_weights = [random.random() for _ in range(size)]
        with pytest.raises(Exception):
            nl = NeuralLayer(
                size=size,
                inputs=inputs,
                weights=weights,
                activation=activation,
                bias=bias,
                bias_weights=bias_weights)
        

    def test_len(self):

        size = random.randint(2, 30)
        inputs = random.randint(1, 30)
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid
        # No bias
        bias = False
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)
        assert len(nl) == size

    def test_getitem(self):

        size = random.randint(2, 30)
        inputs = random.randint(1, 30)
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid
        # No bias
        bias = False
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)

        rand_index = random.randint(0, size-1)

        assert nl[rand_index].w_ij == weights[rand_index]


    def test_setitem(self):

        size = random.randint(2, 30)
        inputs = random.randint(1, 30)
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid
        # No bias
        bias = False
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)

        rand_index = random.randint(0, size-1)

        n = Neuron([0, 0, 0], sigmoid)

        nl[rand_index] = n

        assert nl[rand_index] is n


    def test_solve(self):

        # Non-Bias
        size = random.randint(2, 30)
        inputs = random.randint(1, 30)
        data = [random.randint(0, 100) for _ in range(inputs)]
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid
        bias = False
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)
        result = nl.solve(data)
        
        true_val = feed_forward([weights,], data, activation)

        assert result == true_val[0]

        # With-Bias

        size = random.randint(2, 30)
        inputs = random.randint(1, 30)
        data = [random.randint(0, 100) for _ in range(inputs)] + [1]
        weights = [
            [random.random() for _ in range(inputs)] 
            for _ in range(size)]
        activation = sigmoid
        bias = True
        bias_weights = [random.random() for _ in range(size)]
        nl = NeuralLayer(
            size=size,
            inputs=inputs,
            weights=weights,
            activation=activation,
            bias=bias,
            bias_weights=bias_weights)
        result = nl.solve(data)[:-1]
        
        net = [[w + [b] for w, b in zip(weights, bias_weights)],]
        true_val = feed_forward(net, data, activation)

        assert result == true_val[-1]



class TestNeuralNetwork:
    def test_init(self):

        # No Bias
        n_layers = random.randint(1, 20)
        shape = [random.randint(2, 30) for _ in range(n_layers)]
        data = [random.randint(0, 100) for _ in range(shape[0])]
        weights = [[[random.random() 
            for _ in range(shape[i-1 if i-1 > 0 else 0])]  # For every neuron in previous layer
                for _ in range(layer_size)]                # For every neuron in layer
                    for i, layer_size in enumerate(shape)] # For every layer
        activation = [sigmoid for _ in range(n_layers)]
        use_bias = False
        bias_weights = [[random.random() 
            for _ in range(size)]
                for size in shape]

        ANN = NeuralNetwork(shape,
            weights,
            activation,
            use_bias,
            bias_weights)
        result = ANN.solve(data)

        true_val = feed_forward(weights,
            data,
            activation[0])

        assert result == true_val[-1]

        # With Bias
        n_layers = random.randint(0, 20)
        shape = [random.randint(2, 30) for _ in range(n_layers)]
        data = [random.randint(0, 100) for _ in range(shape[0])]
        weights = [[[random.random() 
            for _ in range(shape[i-1 if i-1 > 0 else 0])]  # For every neuron in previous layer
                for _ in range(layer_size)]                # For every neuron in layer
                    for i, layer_size in enumerate(shape)] # For every layer
        activation = [sigmoid for _ in range(n_layers)]
        use_bias = True
        bias_weights = [[random.random() 
            for _ in range(size)]
                for size in shape]

        net = [[a+[b] 
            for a, b in zip(l_w, l_b)]
                for l_w, l_b in zip(weights, bias_weights)]

        ANN = NeuralNetwork(shape,
            weights,
            activation,
            use_bias,
            bias_weights)
        result = ANN.solve(data)

        true_val = feed_forward(net,
            data,
            activation[0])

        assert result == true_val[-1]


    def test_solve(self):

        pass