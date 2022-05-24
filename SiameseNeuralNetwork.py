# Author: Arjun Viswanathan
# Date: 5/24/22
# Creating a Siamese Neural Network (SNN) architecture using PyTorch.
# Network takes in 2 sets of inputs, and trains on them to give 2 sets of outputs.
# These outputs are then used to compute a distance, and this is passed into a Dense layer to give the output of the SNN

import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, Sigmoid, Linear
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class SiameseNeuralNetwork():
    def __init__(self):
        self.model = Sequential(
            Conv2d(in_channels=1, out_channels=64, kernel_size=(10,10), stride=1),
            ReLU(),
            MaxPool2d(kernel_size=64, stride=2),
            Conv2d(in_channels=64, out_channels=128, kernel_size=(7,7), stride=1),
            ReLU(),
            MaxPool2d(kernel_size=64, stride=2),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(4,4), stride=1),
            ReLU(),
            MaxPool2d(kernel_size=64, stride=2),
            Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=1),
            ReLU(),
            Linear(in_features=256, out_features=1024),
            Sigmoid()
        )

        self.output = Linear(in_features=1024, out_features=1)

    def forward_on_input(self, x):
        return self.model(x)

    def forward(self, x1, x2):
        y1 = self.forward_on_input(x1)
        y2 = self.forward_on_input(x2)
        d = torch.abs(y1 - y2)
        p = self.output(d)
        return p

if __name__ == "__main__":
    nn = SiameseNeuralNetwork()
    print(nn)