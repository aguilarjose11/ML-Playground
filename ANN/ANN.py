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
        y_pred: Union[float, np.ndarray],
        y:      Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

        # len(shape) == 1 represents an array
        if type(y) == type(y_pred) == float or len(y_pred.shape) == 1:
            # Single value passed
            return np.linalg.norm(y_pred - y)
        else:
            return [np.linalg.norm(a - yy) for a, yy in zip(y_pred, y)]



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

        output = [a.copy()]
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

        Implements backpropagation as defined by the following 4 
        equations:

        delta^L    = /\\_a C * sigma'(z^L)
        delta^l    = ( (w^(l+1)_jk)^T*delta^(l+1) ) * sigma'(Z^l)
        dC/db^l_j  = delta^l
        dC/dw^l_jk = a^(l-1)_k * delta^l

        """

        # Test data usage is unimplemented. TODO
        if sgd:
            # Will train using stochastic GD
            pass # TODO: Implement
        else:
            # Will train with regular GD
            for epoch in range(epochs):
                # changes matrices
                del_b = [np.zeros(layer.shape) for layer in self.biases]
                del_w = [np.zeros(layer.shape) for layer in self.weights]

                for X_t, y_t in zip(X, y):
                    # Calculate Output Layer
                    a = self.feedforward(X_t)
                    z_L = np.matmul(self.weights[-1], a[-2]) + self.biases[-1]
                    # numpy's array's * product is the hadamard product by default
                    # Output layer's error
                    delta_L = self.mse_prime(a, y_t) * self.sigmoid_prime(z_L)
                    # Add output changes.
                    del_b_l = [delta_L]
                    del_w_l = [a[-2] * delta_L]

                    # Calculate hidden layers
                    delta_l = delta_L

                    for l in range(len(self.weights[-2::-1]), 0, -1):
                        z_l     = np.matmul(self.weights[l], a[l-1]) + self.biases[l]
                        delta_l = (self.weights[l].transpose()*delta_l) * self.sigmoid_prime(z_l)
                        del_b_l.append(delta_l)
                        del_w_l.append(a[l-1] * delta_l)

                    # To save time, changes were appended to the end.
                    # Thus, Why we need to reverse them so to match.
                    del_b_l.reverse()
                    del_w_l.reverse()
                    for l, b, w in range(zip(del_b_l, del_w_l)):
                        del_b[l] += b
                        del_w[l] += w

                # Calculate averaged modification.
                for l, _, _ in range(zip(del_b, del_w)):
                    del_b[l] /= len(y)
                    del_w[l] /= len(y)

                # Apply step
                for l, b, w, _, _ in range(zip(del_b, del_w, self.weights, self.biases)):
                    self.biases[l]  += eta * b
                    self.weights[l] += eta * w

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
    
