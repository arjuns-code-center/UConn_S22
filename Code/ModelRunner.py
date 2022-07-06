import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import SiameseNeuralNetwork as SNN
import MakeDataset as MD

class MR():
    def __init__(self, starting_features, batchsize, epochs, lr_base, lr_max, train_dset, val_dset, test_dset, train_var, tp):        
        self.train_ec_dl = DataLoader(train_dset, shuffle=True, batch_size=batchsize)
        self.val_dset = val_dset
        self.test_dset = test_dset
        
        # Set the model and training parameters
        self.model = SNN.SNN(starting_features)
        self.max_epochs = epochs

        # Set optimizer and loss function. Using MSE for regression. Have lr_scheduler for adaptive learning 
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr_base)
        self.sch = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=lr_base, max_lr=lr_max, mode='exp_range')
        self.criterion = torch.nn.MSELoss()

        # Set the baseline calculation parameter, and calculate the baselines
        self.var = train_var
        self.trainbase = 0
        self.valbase = 0
        self.testbase = 0
        
        if tp == "xe":
            for i in train_dset.xe:
                self.trainbase += self.criterion(self.var, i/100.0).item() / len(train_dset)
            for i in val_dset.xe:
                self.valbase = self.criterion(self.var, i/100.0).item() / len(val_dset)
            for i in test_dset.xe:
                self.testbase = self.criterion(self.var, i/100.0).item() / len(test_dset)
        else:
            for i in train_dset.Te:
                self.trainbase += self.criterion(self.var, i.float()).item() / len(train_dset)
            for i in val_dset.Te:
                self.valbase = self.criterion(self.var, i.float()).item() / len(val_dset)
            for i in test_dset.Te:
                self.testbase = self.criterion(self.var, i.float()).item() / len(test_dset)

        # Set the training parameter. Xe or Te
        self.tp = tp
        
        # For callback
        self.patience = 0

    def train_and_validate(self):
        trloss = np.array([])
        trbase = np.array([])
        vloss = np.array([])
        vbase = np.array([])
        
        lowest_loss = 100
        lowest_loss_epoch = 1
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_running_loss = 0.0
            val_running_loss = 0.0
            
            # training step: iterate through the batch and obtain the 4 data
            for x, (m1, m2, xe, Te) in enumerate(self.train_ec_dl):   
                self.opt.zero_grad()
                
                if self.tp == "xe":
                    truth = xe/100.0
                else:
                    truth = Te.float()

                # pass 2 sets of inputs into the snn and get p, the output
                output = self.model(m1.float(), m2.float(), self.tp)

                loss = self.criterion(output[:, 0], truth)

                loss.backward()
                self.opt.step()

                train_running_loss += loss.item()

            self.sch.step()                                    # update the learning rate
            
            self.model.eval()
            for i in range(len(self.val_dset)):
                line = self.val_dset[i]

                m1 = line[0]
                m2 = line[1]
                if self.tp == "xe":
                    truth = line[2]/100.0
                else:
                    truth = line[3].float()

                output = self.model(m1.float(), m2.float(), self.tp)
                val_running_loss += self.criterion(output[:, 0], truth).item() / len(val_dset)

            print('Epoch {} | Train Loss: {} | Train Baseline: {} | Val Loss: {} | Val Baseline: {}'.format(
                epoch+1, 
                np.round(train_running_loss / x, 3), 
                np.round(self.trainbase, 3), 
                np.round(val_running_loss, 3), 
                np.round(self.valbase, 3)))
            
            # Callback. If the loss goes up, revert the parameters back to previous epoch. Added patience to stop infinite computation. 
            if val_running_loss <= lowest_loss:
                lowest_loss = val_running_loss
                lowest_loss_epoch = epoch+1
                
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict()}, "D:\\Research\\UConn_ML\\Code\\Checkpoints\\checkpoint.pth")
                
                self.patience = 0
            else:
                if self.patience <= 1:
                    print("Callback to epoch {} | Patience {}/2".format(lowest_loss_epoch, self.patience+1))
                    
                    checkpoint = torch.load("D:\\Research\\UConn_ML\\Code\\Checkpoints\\checkpoint.pth")
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    self.patience = self.patience + 1
                
            # Early stopping. If the train loss goes below a certain value, then we can stop training, preventing overfitting. 
            if train_running_loss <= 17.5:
                print("Early Stop")
                break
            
            trloss = np.append(trloss, train_running_loss)
            trbase = np.append(trbase, train_base_loss)
            vloss = np.append(vloss, val_running_loss)
            vbase = np.append(vbase, val_base_loss)

        try:
            x = np.arange(epoch)
        except ValueError:
            x = np.arange(self.max_epochs)
            
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
        self.model.eval()

        fig, axes = plt.subplots(8, 2)
        fig.set_figheight(40)
        fig.set_figwidth(15)
        row = 0

        with torch.no_grad():
            for i in range(10):
                line = self.test_dset[i]
                m1 = line[0]
                m2 = line[1]
                
                if self.tp == "xe":
                    truth = line[2]/100.0
                else:
                    truth = line[3].float()

                outputs = self.model(m1.float(), m2.float(), self.tp) # f(A,B)
                invouts = self.model(m2.float(), m1.float(), self.tp) # f(B,A)

                test_loss += self.criterion(outputs, truth).item() / len(test_dset)

                x = np.arange(10)

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
                np.round(self.testbase, 3)))

            axes[0, 0].set_title("Residual plots of predicted and actual eutectic proportion Xe")
            axes[0, 1].set_title("Scatter plots of predicted vs actual eutectic proportion Xe")
            plt.show()

        # fig.savefig('D:\\Research\\UConn_ML\\Images\\snn_results_plots.png')

        # Print the values from the last batch processed just for the user to see
        print("f(A,B): \n", outputs.flatten())
        print("\n")
        print("f(B,A): \n", invouts.flatten())
        print("\n")
        l = min(len(outputs), len(invouts))
        # should all be 1 or close to 1 to show that f(A,B) = 1 - f(B,A)
        print("f(A,B) + f(B,A): \n", outputs[0:l].flatten() + invouts[0:l].flatten())
        print("\n")
        print("Original Values: \n", truth)
        print("\n")
        print("Predicted Values: \n", outputs.flatten())