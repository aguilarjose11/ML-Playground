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
        z: np.ndarray) -> Union[np.ndarray, float]:
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
        -------- = -------- * --- (y_pred - y)^2 = 
        d y_pred   d y_pred    2

                              d
        = (y_pred - y) * -------- * (y_pred - y) = (y_pred - y)
                           d y_pred
        """

        return y_pred - y




    def feedforward(self, 
        a: np.ndarray) -> List[np.ndarray]:
        """Network solve function
        
        Feed forward information. Will return the
        activation for every layer.
        
        Input must be (n, 1) shape. 
        
        parameters
        ----------
        a: np.ndarray
            Input data
        """

        assert len(a.shape) == 2, f"Invalid shape {a.shape}."
        output = [a.copy()]
        for b_l, w_l in zip(self.biases, self.weights):
            z_l = np.matmul(w_l, a) + b_l
            a_l = self.sigmoid(z_l)
            output.append(a_l)
            a = a_l # feed to next layer
        return output

    
    def _feedforward(self, 
        a: np.ndarray) -> List[np.ndarray]:
        """Network solve function
        
        Feed forward information. Will return the activation for 
        every layer as well as its weighted sum. This function is 
        for exclusive use in the backpropagation algorithm
        
        Input must be (n, 1) shape. 
        
        parameters
        ----------
        a: np.ndarray
            Input data
        """

        assert len(a.shape) == 2, f"Invalid shape {a.shape}."
        output = [a.copy()]
        z = []
        for b_l, w_l in zip(self.biases, self.weights):
            z_l = np.matmul(w_l, a) + b_l
            a_l = self.sigmoid(z_l)
            z.append(z_l)
            output.append(a_l)
            a = a_l # feed to next layer
        return output, z



    def backprop(self,
        X: List[Tuple[float, ...]],
        y: List[float],
        epochs: int=50,
        eta: float=1.,
        X_test: List[Tuple[float, ...]]=None,
        y_test: List[float]=None,
        sgd: bool=True):

        for epoch in range(epochs):
            for datum, label in zip(X, y):
                a, z = self._feedforward(datum)
                # Backpropagate through output layer
                
                del_k = self.sigmoid(z[-1])*(1-self.sigmoid(z[-1]))*(label - self.sigmoid(z[-1]))
                self.weights[-1] += eta * np.matmul(del_k, a[-2].T)
                self.biases[-1] += eta * del_k

                for l in range(len(self.biases)-2, -1, -1):
                    del_h = self.sigmoid(z[l])*(1-self.sigmoid(z[l]))*np.matmul(self.weights[l+1].transpose(), del_k)
                    self.weights[l] += eta * np.matmul(del_h, a[l].T)
                    self.biases[l] += eta * del_h
                    del_k = del_h

            print(f"Epoch {epoch} completed.")



    def train(self,
        X: List[Tuple[float, ...]],
        y: List[float],
        epochs: int=50,
        eta: float=1.,
        X_test: List[Tuple[float, ...]]=None,
        y_test: List[float]=None,
        sgd: bool=True,
        mini_batch_size: int=1) -> None:
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

                for datum, target in zip(X, y):
                    del_b_t = []
                    del_w_t = []
                    # Forwardpropagate
                    a, z = self._feedforward(datum)
                    # Output cost gradient, conserves dimensions of output
                    grad_C_L = self.mse_prime(a[-1], target)
                    # output Net (weighted sum)
                    z_L = z[-1]
                    # sigma derivative, conserves dimensions of z
                    sigma_L  = self.sigmoid_prime(z_L)
                    # Double check for errors
                    assert grad_C_L.shape == sigma_L.shape, "Problem with hadamard product!"
                    # find output layer error. Hadamard product equivalent.
                    del_L = grad_C_L * sigma_L
                    # Find output bias changes
                    del_b_t.append(del_L)
                    # Find output weights changes.
                    # Dimensions shall be same as weight matrix of layer.
                    del_w_t.append(np.matmul(del_L, a[-2].T))
                    # Prepare for backpropagation
                    del_l = del_L

                    # For every layer...
                    for l in range(len(self.weights[:-2]), -1, -1):
                        # Because of feedworward function, x is included,
                        # Thus shifting activations to the right and making
                        # a^l-1 => a^l
                        z_l = z[l]
                        del_l = np.matmul(self.weights[l+1].T, del_l) * self.sigmoid_prime(z_l)
                        del_b_t.append(del_l)
                        del_w_t.append(np.matmul(del_l, a[l].T))
                    
                    # Add gradients for current training instance.
                    del_b_t.reverse()
                    del_w_t.reverse()
                    for l, (b, w) in enumerate(zip(del_b_t, del_w_t)):
                        del_b[l] += -b
                        del_w[l] += -w # Do numpy's matrices add as expected?
                
                #print(f"before: Bias: {self.biases} Weights: {self.weights}")
                for l, (b, w) in enumerate(zip(del_b, del_w)):
                    # gradients are averaged out. Optional tho.
                    self.biases[l]  += (b / len(y)) * eta
                    self.weights[l] += (w / len(y)) * eta
                #print(f"after: Bias: {self.biases} Weights: {self.weights}")

                print(f"Epoch {epoch} completed")
                


              
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
    
