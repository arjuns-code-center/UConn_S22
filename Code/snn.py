# Author: Arjun Viswanathan
# Date: 5/24/22
# Creating a Siamese Neural Network (SNN) architecture using PyTorch.
# Network takes in 2 sets of inputs, and trains on them to give 2 sets of outputs.
# These outputs are then used to compute a distance, and this is passed into a Dense layer to give the output of the SNN

import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, Sigmoid, Linear
from torchvision import datasets as dts
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

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

# Testing the SNN with MNIST to see what happens
class LoadData():
    def __init__(self):
        self.train = dts.MNIST(
            root= 'data',
            train= True,
            transform= ToTensor(),
            download= True,
        )

        self.test = dts.MNIST(
            root= 'data',
            train= False,
            transform= ToTensor()
        )

        self.train_loader = DataLoader(self.train, batch_size=100, shuffle=True)
        self.test_loader = DataLoader(self.test, batch_size=100, shuffle=True)

    def getTrain(self):
        return self.train_loader

    def getTest(self):
        return self.test_loader

# Training the SNN on the test data (MNIST) loaded
class TrainSNN():
    def __init__(self):
        self.nn = SiameseNeuralNetwork()
        self.ld = LoadData()
        self.train()

    # TODO: Fix the training after gaining more knowledge
    def train(self):
        loss = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.nn.model.parameters(), lr=0.05)

        for epoch in range(10):
            for x, (images, labels) in enumerate(self.ld.getTrain()):
                images = images.reshape(28, 28)

                output = self.nn.forward(images, images)
                losses = loss(output, labels)

                opt.zero_grad()
                losses.backward()
                opt.step()

            if((x+1) % 100 == 0):
                print(f'Epochs [{epoch + 1}/{10}], Step[{x+1}/{len(self.ld.getTrain())}], Losses: {losses.item():.4f}')

if __name__ == "__main__":
    TrainSNN()