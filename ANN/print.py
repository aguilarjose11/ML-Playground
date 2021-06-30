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
                    del_b[l] += b
                    del_w[l] += w # Do numpy's matrices add as expected?
            
            #print(f"before: Bias: {self.biases} Weights: {self.weights}")
            for l, (b, w) in enumerate(zip(del_b, del_w)):
                # gradients are averaged out. Optional tho.
                self.biases[l]  += (b / len(y)) * eta
                self.weights[l] += (w / len(y)) * eta
            #print(f"after: Bias: {self.biases} Weights: {self.weights}")

            print(f"Epoch {epoch} completed")
            

