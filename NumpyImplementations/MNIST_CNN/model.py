import numpy as np
from layers import *

BETA1 = 0.9
BETA2 = 0.98
EPSILON = 1e-7
LR = 1e-3

# input shape: (batch_size, 28, 28, 1)
class CNN():
    def __init__(self):
        self.conv1 = Conv2D(kernel_size=3, 
                            num_input_channels=1, 
                            num_output_channels=32, 
                            beta1=BETA1, beta2=BETA2, 
                            epsilon=EPSILON, 
                            learning_rate=LR)
        self.pool1 = Pooling2D(filter_size=2, pooling_method='max')
        self.conv2 = Conv2D(kernel_size=3, 
                            num_input_channels=32, 
                            num_output_channels=64, 
                            beta1=BETA1, beta2=BETA2, 
                            epsilon=EPSILON, 
                            learning_rate=LR)
        self.pool2 = Pooling2D(filter_size=2, pooling_method='max')
        self.flatten = Flatten()
        self.linear1 = FFN(in_features=1600, 
                           out_features=128, 
                           beta1=BETA1, 
                           beta2=BETA2, 
                           epsilon=EPSILON, 
                           learning_rate=LR)
        self.linear2 = FFN(in_features=128, 
                           out_features=10, 
                           beta1=BETA1, 
                           beta2=BETA2, 
                           epsilon=EPSILON, 
                           learning_rate=LR)
        self.softmax = Softmax()

    def forward(self, X):
        Z1, A1 = self.conv1.forward(X)
        A2 = self.pool1.forward(A1)
        Z3, A3 = self.conv2.forward(A2)
        A4 = self.pool2.forward(A3)
        A5 = self.flatten.forward(A4)
        Z6, A6 = self.linear1.forward(A5)
        A7 = self.linear2.forward(A6, use_relu=False)
        A8 = self.softmax.forward(A7)
        return (Z1, A1), (A2,), (Z3, A3), (A4,), (A5,), (Z6, A6), (A7,), (A8,)

    def backward(self, out_tuples, X, Y):
        (Z1, A1), (A2,), (Z3, A3), (A4,), (A5,), (Z6, A6), (A7,), (A8,) = out_tuples
        dA7 = self.softmax.backward(A8, Y)
        dAprev = self.linear2.backward(dA=dA7, A=A6, used_relu=False)
        dAprev = self.linear1.backward(dA=dAprev, A=A5, Z=Z6)
        dAprev = self.flatten.backward(dAprev, A4)
        dAprev = self.pool2.backward(dAprev, A3, A4)
        dAprev = self.conv2.backward(A2, Z3, dAprev)
        dAprev = self.pool1.backward(dAprev, A1, A2)
        dAprev = self.conv1.backward(X, Z1, dAprev)


