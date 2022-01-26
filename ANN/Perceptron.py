"""Perceptron Learning Algorithms

Developed following Valiant's description of a PAC learnable 
Perceptron (PAC Learning, Ch 3.7)
"""

import numpy as np
from typing import List, Tuple

class Perceptron:
    """Perceptron class
    
    Defines a perceptron model as described by Valiant's PAC book

    Member Variables
    ----------------
    perceptron: List
    - Perceptron's weights
    bias: List # Unimplemented
    - Perceptron's biases

    Functions
    ---------
    __init__: -> Perceptron
    - Initializer
    feedforward: -> List
    - Feeds data through the perceptron to produce a generalization 
      (prediction)
    
    """

    perceptron: np.ndarray
    
    def __init__(self, 
        input: int=2,
        output: int=1,
        zeros: bool=True):


        if not zeros:
            self.perceptron = np.random.randn(output, input)
        else:
            self.perceptron = np.array([np.zeros(input) for _ in range(output)])
        
        self.r = 0


    def feedforward(self, 
        datum: List[float]) -> List[int]:
        """Perceptron feed forward function

        We can percieve this process as the activation function for
        some neuron that has some weights. The weighted sum typically
        see in regular neural networks will be seen as the dot 
        product of these weights by the input values. With this
        analogy in mind, we can see the activation function of this
        neuron to be some special instance of the step function where
        a cut off of 0 is used. Additionally, this activation can be
        seen as an indicator function for the membership of the 
        predicted label.
        """
        
        # Can also be represented by the dot product.
        datum = np.array(datum).transpose()
        # This function represents an activation function
        classify = np.vectorize(lambda a, r: 1 if a > r else 0)
        return classify(np.matmul(self.perceptron, datum), self.r)

    def train(self, X: np.ndarray, y: np.ndarray):
        """data shall be a matrix of training instances."""

        for datum, label in zip(X, y):
            
            y_pred = self.feedforward(datum)
            print(f"label: {label} vs pred: {y_pred}")
            if y_pred != label:
                # Missclasified
                for j in range(len(self.perceptron)):
                    if label:
                        print(f"Pred: {y_pred} - actual {label} - false positive")
                        self.perceptron[j] += datum
                    else:
                        print(f"Pred: {y_pred} - actual {label} - false negative")
                        self.perceptron[j] -= datum

class PocketPerceptron:
        def __init__(self, 
            input: int=2,
            output: int=1,
            zeros: bool=True,):


            self.zeros = zeros
            self.input = input
            self.output = output
            self.pocket = np.array([])
            self.r = 0

        def score(self, X: List[float], Y: List[float]) -> Tuple[int, int]:
            num_ok = num_nok = 0
            for x, y in zip(X, Y):
                y_pred = self.feedforward(x)
                if y_pred == y:
                    num_ok += 1
                else:
                    num_nok += 1
            return num_ok, num_nok


        def feedforward(self, 
            datum: List[float]) -> List[int]:
            """Perceptron feed forward function

            We can percieve this process as the activation function for
            some neuron that has some weights. The weighted sum typically
            see in regular neural networks will be seen as the dot 
            product of these weights by the input values. With this
            analogy in mind, we can see the activation function of this
            neuron to be some special instance of the step function where
            a cut off of 0 is used. Additionally, this activation can be
            seen as an indicator function for the membership of the 
            predicted label.
            """
            
            # Can also be represented by the dot product.
            datum = np.array(datum).transpose()
            # This function represents an activation function
            classify = np.vectorize(lambda a, r: 1 if a > r else 0)
            return classify(np.matmul(self.perceptron, datum), self.r)

        def train(self, X: np.ndarray, Y: np.ndarray):
            if not self.zeros:
                self.perceptron = np.random.randn(self.output, self.input)
            else:
                self.perceptron = np.array([np.zeros(self.input) for _ in range(self.output)])
            run_p = run_w = num_okW = 0
            for x, y in zip(X, Y):
                if self.feedforward(x) == y:
                    run_p += 1
                    if run_p > run_w:
                        num_ok, num_nok = self.score(X, Y)
                        if num_ok > num_okW:
                            self.pocket = self.perceptron
                            run_w = run_p
                            num_okW = num_ok
                            if num_nok == 0:
                                #done
                                break
                else:
                    for j in range(len(self.perceptron)):
                        if y:
                            self.perceptron[j] += x[j]
                        else:
                            self.perceptron[j] -= x[j]

                    run_w = 0

