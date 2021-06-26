import numpy as np
from typing import List, Tuple, Any, Union

# Code from "Neural Networks and Deep Learning" book

# Neural Network Master class
class Network:
    """Feeddorward Neural Network.

    Basic feedforward neural network that creates
    a network based on the given list of neurons
    per layer. The notation is as follows:
    w_jk: Synapse between kth neuron at Pre-synaptic 
    layer and jth neuron at Post-synaptic layer.
    b_j: bias for jth post-synaptic layer.
    a_l: Activation of lth layer.

    Variables
    ---------
    num_layers: int
        Number of layers in neural network. It 
        includes input and output layers.
    sizes: List[int]
        List of neurons per layer. len(sizes) ==
        num_layers.
    biases: List[np.ndarray]
        List of biases for every neuron at every
        layer except input layer. biases[l][j] will
        select the bias for layer l, neuron j.
    weights: List[np.ndarrray]
        List of matrices of weights for every synapse
        across each neighbooring layer. 
        weights[l][j][k] will select the weight for
        the synapse between pre-synaptic kth neuron
        and post-synaptic jth neuron based on post-
        synaptic lth layer.
    """

    # Number of layers. Aka. Network depth
    num_layers: int
    # Network design
    sizes: List[int]
    # List of biases. One per neuron. Not per layer
    biases: List[np.ndarray]
    # List of weights.
    weights: List[np.ndarray]

    def __init__(self, 
        sizes: List[int]) -> None:
        """Create neural network.
        
        Parameters
        ----------
        sizes: List[int]
            List of neurons per layer. It shall not
            include biases.
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(neurons, 1) 
            for neurons in sizes[1:]]
        self.weights = [np.random.randn(neurons, synapses)
            for neurons, synapses in 
                zip(sizes[1:], sizes[:-1])]


    def sigmoid(self,
        z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function
        
        Applies the mathematical special softmax
        sigmoid function that follows the following
        formula:
                     1
        sigmoid = ---------
                   1 + e^(-z)

        Note that this function will vectorize
        arrays.

        parameters
        ----------
        z: np.ndarray
            Input integral.
        """

        return 1. / (1.+np.exp(-z))

    
    def sigmoid_prime(self,
        z: np.ndarray) -> float:
        """Derivative of sigmoid function.
        
        This derivative is defined as follows:
        let sig(z) be the sigmoid function:
                    1
        sig(z) = --------                                         (1)
                 1 + e^(-z)

        Following the quotient rule, we find:

                           -e^(-z)
        (d/dz) sig(z) = --------------                            (2)
                        [1 + e^(-z)]^2

        Which can be split as:
                            1          -e^(-z)
        (d/dz) sig(z) = ----------- * ----------                  (3)
                         1 + e^(-z)   1 + e^(-z)

                     -e^(-z)
        = sig(z) * -----------
                    1 + e^(-z)

        We note that:

          -e^(-z)     1 + e^(-z)      -1
        ----------- = ----------  + ----------                    (4)
         1 + e^(-z)   1 + e^(-z)    1 + e^(-z)

        = 1 - sig(z)

        hence:

        (d/dz) sig(z) = sig(z)(1 - sig(z))                        (5)
        """

        return self.sigmoid(z)*(1 - self.sigmoid(z))


    def mse(self, 
        y_pred: np.ndarray, 
        y:      np.ndarray):
        """Mean Squared Error.
        
        Uses the Mean Squared Error (MSE):
        
                         1  ___ n
                        --- \\         ||y_i-y_hat_i||^2          (1)
        MSE(y, y_hat) = 2n  //_ i = 1

        For simplicity, this function will find the MSE for an 
        individual predicted, label pair. Thus, the formula is:
      
                         1
        MSE(y, y_hat) = ---  ||y - y_hat||^2                      (2)
                         2

        Where y, y_hat are elements of R^n. Thus, from an ANN point
        of view, n is the number of output neurons. Note that the 1/2
        added to the formula is there for derivation simplicity.

        Note that ||x|| is the euclidean norm/distance:
                 _    ________________________
                  |  / ___ n
        ||x|| =   | /  \\      (x_i)^2                            (3)
                  |/   //_ i=1
        """         

        n = len(y)
        return (1/2) * np.linalg.norm(y_pred-y)**2


    def mse_prime(self,
        y_pred: np.ndarray,
        y:      np.ndarray) -> float:
        """MSE function derivative
        
        The function derivative of this MSE uses a "sweeter" version
        of MSE as described in the MSE function. This derivation
        is defined as follows:

         d MSE        d        1
        -------- = -------- * --- ||y_pred - y||^2 = 
        d y_pred   d y_pred    2

                              d
        = ||y_pred - y|| * -------- * ||y_pred - y|| = ||y_pred - y||
                           d y_pred
        """

        return np.linalg.norm(y_pred - y)



    def feedforward(self, 
        a: np.ndarray) -> List[np.ndarray]:
        """Network solve function
        
        Feed forward information. Will return the
        activation for every layer.
        
        parameters
        ----------
        a: np.ndarray
            Input data
        """

        output = []
        for b_l, w_l in zip(self.biases, self.weights):
            z_l = np.matmul(w_l, a) + b_l
            a_l = self.sigmoid(z_l)
            output.append(a_l)
            a = a_l # feed to next layer
        return output

    
    def train(self,
        X: List[Tuple[float, ...]],
        y: List[float],
        epochs: int=50,
        eta: float=0.01,
        X_test: List[Tuple[float, ...]]=None,
        y_test: List[float]=None,
        sgd: bool=True,
        mini_batch_size: int=1):
        """ ANN training

        This ANN will utilize backpropagation. Both stochastic and
        non-stochastic options are included.

        Parameters
        ----------
        X: List[Tuple[float, ...]]
        - Training data. Shall be a list of instances
        y: List[float]
        - Training labels.
        epochs: int=50
        - Number of steping epochs.
        eta: float=0.01
        - Size of gradient step.
        X_test: None or List=None
        - Testing data.
        y_test: None or List=None
        - Testing data's labels
        mini_batch_size: int=1
        - Size of stochastic gradient descent's batches
        sgd: bool=True
        - Flag to use stochastic gradient descent or regular descent.
        """

        # Test data usage is unimplemented. TODO
        if sgd:
            # Will train using stochastic GD
            pass # TODO: Implement
        else:
            # Will train with regular GD
            for epoch in range(epochs):
                for X_t, y_t in zip(X, y):
                    a = self.feedforward(X_t)

                    z_L = np.matmul(self.weights[-1], a[-2]) + self.biases[-1]
                    # numpy's array's * product is the hadamard product by default
                    delta_L = 
                    for l in 


    def gradient_descent(self,
        data: List[Any],
        eta: float):
        """Apply Gradient Descent

        Applies gradient descent algorithm using
        backpropagation:

        w_ij(t+1) = w_ij(t) - eta*/\\w_ij(d)/|data|
        - The weight modification will be a 
          modification that shall be the average
          of the gradient changes for every single
          training instance.
        
        """

        # Initial Bias gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # Initial weight gradients
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in data:
            # Find changes for  bias and weights
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        
        self.weights = [w-(eta/len(data))*nw 
            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(data))*nb
            for b, nb in zip(self.biases, nabla_b)]
    

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)