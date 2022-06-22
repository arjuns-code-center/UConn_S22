import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import SiameseNeuralNetwork as SNN

# TODO: Implement Early Stopping to better tune the parameters. Have defined some variables for that.

class MR():
    def __init__(self, starting_features, batchsize, epochs, lr_base, lr_max, train_dset, val_dset, test_dset, trainbase, valbase, testbase, tp):
        # For Xe
        self.train_ec_dl = DataLoader(train_dset, shuffle=True, batch_size=batchsize)
        self.val_ec_dl = DataLoader(val_dset, shuffle=True, batch_size=batchsize)
        self.test_ec_dl = DataLoader(test_dset, shuffle=True, batch_size=batchsize)

        # Set the model and training parameters
        self.model = SNN.SNN(starting_features)
        self.num_epochs = epochs

        # Set optimizer and loss function. Using MSE for regression. Have lr_scheduler for adaptive learning 
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr_base)
        self.sch = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=lr_base, max_lr=lr_max, mode='exp_range', verbose=True)
        self.criterion = torch.nn.MSELoss()
        
        self.trainbase = trainbase
        self.valbase = valbase
        self.testbase = testbase
        
        self.tp = tp
        self.modelparams = None

    def train_and_validate(self):
        trloss = np.array([])
        trbase = np.array([])
        vloss = np.array([])
        vbase = np.array([])
        
        lowest_loss = 100
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_running_loss = 0.0
            train_base_loss = 0.0
            val_running_loss = 0.0
            val_base_loss = 0.0

            # training step: iterate through the batch and obtain the 4 data
            for x, (m1, m2, xe, Te) in enumerate(self.train_ec_dl):   
                self.opt.zero_grad()
                
                if self.tp == "xe":
                    truth = xe/100.0
                else:
                    truth = Te

                # pass 2 sets of inputs into the snn and get p, the output
                output = self.model(m1.float(), m2.float(), self.tp)

                loss = self.criterion(output[:, 0], truth)

                base = torch.full((len(truth),), self.trainbase)     # create same value array
                base_loss = self.criterion(base, truth)            # obtain baseline loss

                loss.backward()
                self.opt.step()

                train_running_loss += loss.item()
                train_base_loss += base_loss.item()

            self.sch.step()
            self.model.eval()
            for v, (m1, m2, xe, Te) in enumerate(self.val_ec_dl):
                if self.tp == "xe":
                    truth = xe/100.0
                else:
                    truth = Te

                output = self.model(m1.float(), m2.float(), self.tp)
                val_running_loss += self.criterion(output[:, 0], truth).item()

                base = torch.full((len(truth), ), self.valbase)
                val_base_loss += self.criterion(base, truth).item()
                
                # if val_running_loss <= lowest_loss:
                #     self.modelparams = torch.nn.ParameterList(self.model.parameters())
                #     lowest_loss = val_running_loss
                #
                # if val_running_loss > lowest_loss:
                #     self.model.parameters() = self.modelparams
                    


            print('Epoch {} | Train Loss: {} | Train Baseline: {} | Val Loss: {} | Val Baseline: {}'.format(
                epoch+1, 
                np.round(train_running_loss, 3), 
                np.round(train_base_loss, 3), 
                np.round(val_running_loss, 3), 
                np.round(val_base_loss, 3)))

            trloss = np.append(trloss, train_running_loss)
            trbase = np.append(trbase, train_base_loss)
            vloss = np.append(vloss, val_running_loss)
            vbase = np.append(vbase, val_base_loss)

        x = np.arange(self.num_epochs)
        plt.figure(1)
        plt.plot(x, trloss, label="Train Running Loss", c="blue")
        plt.plot(x, trbase, label="Train Baseline Loss", c="red")
        plt.title("Graph of Training Loss Against a Baseline")
        plt.legend(loc="upper right")
        plt.show()

        plt.figure(2)
        plt.plot(x, vloss, label="Val Running Loss", c="blue")
        plt.plot(x, vbase, label="Val Baseline Loss", c="red")
        plt.title("Graph of Validation Loss Against a Baseline")
        plt.legend(loc="upper right")
        plt.show()

    def test_plot_stats(self):
        test_loss = 0.0
        test_baseline = 0.0
        self.model.eval()

        fig, axes = plt.subplots(8, 2)
        fig.set_figheight(40)
        fig.set_figwidth(15)
        row = 0

        with torch.no_grad():
            for y, (m1, m2, xe, Te) in enumerate(self.test_ec_dl):
                if self.tp == "xe":
                    truth = xe/100.0
                else:
                    truth = Te

                outputs = self.model(m1.float(), m2.float(), self.tp)
                invouts = self.model(m2.float(), m1.float(), self.tp)

                test_loss += self.criterion(outputs[:, 0], truth).item()

                base = torch.full((len(truth),), self.testbase)
                test_baseline += self.criterion(base, truth).item()

                x = np.arange(len(xe))

                axes[row, 0].scatter(x, outputs.detach().numpy() - truth[np.newaxis].numpy().T, c="red")
                axes[row, 0].plot(x, np.zeros((len(truth.numpy()),)), c="green", label="0 Point")
                axes[row, 0].set(xlabel="Batch Data Points", ylabel="Residuals")
                axes[row, 0].legend(loc="upper right")

                axes[row, 1].scatter(truth.numpy(), outputs.detach().numpy(), c="green")
                axes[row, 1].plot(truth.numpy(), truth.numpy(), label="Accuracy Line")
                axes[row, 1].set(xlabel="Actual Xe", ylabel="Predicted Xe")
                axes[row, 1].legend(loc="upper right")

                plt.ylim([0, 1])

                row += 1
                if row == 8:
                    break

            print('Test Loss: {} | Test Baseline: {}\n'.format(
                np.round(test_loss, 3), 
                np.round(test_baseline, 3)))

            axes[0, 0].set_title("Residual plots of predicted and actual eutectic proportion Xe")
            axes[0, 1].set_title("Scatter plots of predicted vs actual eutectic proportion Xe")
            plt.show()

        # fig.savefig('D:\\Research\\UConn_ML\\Images\\snn_results_plots.png')
        
        # should all be 1 or close to 1 to show that f(A,B) = 1 - f(B,A)
        print("f(A,B): \n", outputs.flatten())
        print("\n")
        print("f(B,A): \n", invouts.flatten())
        print("\n")
        l = min(len(outputs), len(invouts))
        print("f(A,B) + f(B,A): \n", outputs[0:l].flatten() + invouts[0:l].flatten())
        print("\n")
        print("Original Values: \n", truth)
        print("\n")
        print("Predicted Values: \n", outputs.flatten())