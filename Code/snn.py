# Author: Arjun Viswanathan
# Date Created: 5/24/22
# Creating a Siamese Neural Network (SNN) architecture using PyTorch.
# Network takes in 2 sets of inputs, and trains on them to give 2 sets of outputs.
# These outputs are then used to compute a distance, and this is passed into a Dense layer to give the output of the SNN

'''
Log:
5/24: Created file and started the SNN construction. Added in data loading and training to test out with MNIST
database but there were errors in setting up the training.
5/26: After consulting some sample code from a friend, fixed the training code and data loaders. Tested it and it
works, just not very well. Current train accuracy is 10.5% and test accuracy is 11.37%. Will need to adjust parameters
'''

import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear
import torch.nn.functional as F
from torchvision import datasets as dts
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

class SiameseNeuralNetwork(Module):
    def __init__(self):
        super(SiameseNeuralNetwork, self).__init__()
        # First L-2 layers have convolutions followed by max pooling and activation ReLU
        # The L-1 layer is a Dense layer which will give a feature vector with activation Sigmoid
        # The L layer (output layer) will then compute the classification
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding='same')
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
        self.pool2 = MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.pool3 = MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv4 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')

        self.fc1 = Linear(in_features=6400, out_features=1024)
        self.fc2 = Linear(in_features=1024, out_features=10)

    def forward_on_input(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        x = x.flatten(start_dim=1)

        x = torch.sigmoid(self.fc1(x))
        return x

    def forward(self, x1, x2):
        y1 = self.forward_on_input(x1)
        y2 = self.forward_on_input(x2)
        d = torch.abs(y1 - y2)
        p = self.fc2(d)
        return p

# Testing the SNN with MNIST to see what happens
class LoadData():
    def __init__(self):
        self.batch_size = 64

        # set variable transform to covert MNIST data into a Tensor
        transform = transforms.Compose([transforms.ToTensor()])

        # download the MNIST database into a training and validation set, converting to a Tensor as we download
        self.train = dts.MNIST(
            root= './data',
            train= True,
            download= True,
            transform=transform
        )

        self.test = dts.MNIST(
            root= './data',
            train= False,
            download= True,
            transform= transform
        )

        # Use DataLoader to build convenient data loaders (iterators) which feed batches of data to the model
        self.train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=True)

    def imshow(self, imgs):
        # create 5 image plots showing MNIST images
        for img in range(5):
            plt.figure(img)
            plt.imshow(imgs[img], cmap='gray')
        plt.show()

    def dimensions(self):
        for images, labels in self.train_loader:
            print("Image Batch Dimensions: ", images.shape)
            print("Single Image Dimensions: ", images[0].shape)
            print("Labels Batch Dimensions: ", labels.shape)
            break

    def classes(self):
        print("Train Dataset # Classes: ", len(self.train.classes))
        print("Test Dataset # Classes: ", len(self.test.classes))

# Training the SNN on the test data (MNIST) loaded. Currently it will be same set of images into both twins so
# everything should just be classified as the same in the results. Later there will be 2 sets of imports.
class TrainSNN():
    def __init__(self):
        self.model = SiameseNeuralNetwork()
        self.ld = LoadData()

        # If a GPU is available, then send it to that GPU rather than train on CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, num_epochs):
        # set the loss and optimizer
        loss = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.model = self.model.to(self.device)

        for epoch in range(num_epochs):
            train_running_loss = 0.0
            train_accuracy = 0.0
            self.model = self.model.train()

            # training step: iterate through the batch and get the images and labels at each x
            for x, (images, labels) in enumerate(self.ld.train_loader):
                # sending images and labels to device (GPU or CPU)
                images = images.to(self.device)
                labels = labels.to(self.device)

                # pass 2 sets of inputs into the snn and gives p, the output
                output = self.model(images, images)
                losses = loss(output, labels)

                opt.zero_grad()
                losses.backward()
                opt.step()

                train_running_loss += losses.item()
                train_accuracy += self.accuracy(output, labels, self.ld.batch_size)

            self.model.eval()
            print('Epoch %d | Loss: %.4f | Train Accuracy: %.2f'%(epoch+1, train_running_loss / x, train_accuracy / x))

    def test(self):
        test_accuracy = 0.0
        for y, (images, labels) in enumerate(self.ld.test_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images, images)
            test_accuracy += self.accuracy(outputs, labels, self.ld.batch_size)
        print('Test Accuracy: %.2f'%(test_accuracy / y))

    # Compute the accuracy of the model at each epoch
    def accuracy(self, output, target, batch_size):
        corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()

if __name__ == "__main__":
    train_snn = TrainSNN()
    load_data = LoadData()
    snn = SiameseNeuralNetwork()

    # see the imported data dimensions for train_loader
    load_data.dimensions()

    # get number of classes in the training and testing loaders
    load_data.classes()

    # test out the imshow() function to explore the imported data
    dataiter = iter(load_data.train_loader)
    images, labels = dataiter.__next__()
    # show images
    load_data.imshow(images.squeeze())

    # print the model summary
    print(snn)

    # start training
    train_snn.train(10)
    # test once done
    train_snn.test()