import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import SiameseNeuralNetwork as SNN

class MR():
    def __init__(self, starting_features, batchsize, epochs, lr_base, lr_max, train_dset, val_dset, test_dset, train_var, tp):        
        self.train_ec_dl = DataLoader(train_dset, shuffle=True, batch_size=batchsize, drop_last=True)
        self.train_dset = train_dset
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
            for i in train_dset:
                self.trainbase += self.criterion(self.var, i[2]/100.0).item() / len(train_dset)
            for i in val_dset:
                self.valbase += self.criterion(self.var, i[2]/100.0).item() / len(val_dset)
            for i in test_dset:
                self.testbase += self.criterion(self.var, i[2]/100.0).item() / len(test_dset)
        else:
            for i in train_dset:
                self.trainbase += self.criterion(self.var, i[3].float()).item() / len(train_dset)
            for i in val_dset:
                self.valbase += self.criterion(self.var, i[3].float()).item() / len(val_dset)
            for i in test_dset:
                self.testbase += self.criterion(self.var, i[3].float()).item() / len(test_dset)

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
                
            train_running_loss = train_running_loss / x
            self.sch.step()                                    # update the learning rate
            
            self.model.eval()
            for line in self.val_dset:
                m1 = line[0]
                m2 = line[1]
                
                if self.tp == "xe":
                    truth = line[2]/100.0
                else:
                    truth = line[3].float()

                output = self.model(m1.float(), m2.float(), self.tp)
                val_running_loss += self.criterion(output[:, 0], truth).item()
            
            val_running_loss = val_running_loss / len(self.val_dset)
            print('Epoch {} | Train Loss: {} | Train Baseline: {} | Val Loss: {} | Val Baseline: {}'.format(
                epoch+1, 
                np.round(train_running_loss, 6), 
                np.round(self.trainbase, 6), 
                np.round(val_running_loss, 6), 
                np.round(self.valbase, 6)))
            
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
            if train_running_loss <= 0.065:
                print("Early Stop")
                break
            
            trloss = np.append(trloss, train_running_loss)
            trbase = np.append(trbase, self.trainbase)
            vloss = np.append(vloss, val_running_loss)
            vbase = np.append(vbase, self.valbase)
            
        x = np.arange(epoch)
            
        plt.figure(1)
        plt.plot(x, trloss[0:epoch], label="Train Running Loss", c="blue")
        plt.plot(x, trbase[0:epoch], label="Train Baseline Loss", c="red")
        plt.title("Graph of Training Loss Against a Baseline")
        plt.legend(loc="upper right")
        plt.show()

        plt.figure(2)
        plt.plot(x, vloss[0:epoch], label="Val Running Loss", c="blue")
        plt.plot(x, vbase[0:epoch], label="Val Baseline Loss", c="red")
        plt.title("Graph of Validation Loss Against a Baseline")
        plt.legend(loc="upper right")
        plt.show()

    def test_plot_stats(self):
        outputs = np.array([]).astype(float)
        invouts = np.array([]).astype(float)
        truths = np.array([]).astype(float)
        
        test_loss = 0.0
        self.model.eval()

        numplots = 5
        fig, axes = plt.subplots(numplots, 2)
        fig.set_figheight(40)
        fig.set_figwidth(15)

        with torch.no_grad():
            for line in self.test_dset:
                m1 = line[0]
                m2 = line[1]
                
                if self.tp == "xe":
                    truth = line[2]/100.0
                else:
                    truth = line[3].float()
                truths = np.append(truths, truth[np.newaxis].numpy().T)

                output = self.model(m1.float(), m2.float(), self.tp) # f(A,B)
                invout = self.model(m2.float(), m1.float(), self.tp) # f(B,A)
                
                outputs = np.append(outputs, output.detach().numpy())
                invouts = np.append(invouts, invout.detach().numpy())

                test_loss += self.criterion(output, truth).item() / len(self.test_dset)

        l = 25
        pred = 0
        succ = 1
        x = np.arange(l)
        
        for row in range(numplots):
            axes[row, 0].scatter(x, outputs[pred*l:succ*l] - truths[pred*l:succ*l], c="red")
            axes[row, 0].plot(x, np.zeros((l,)), c="green", label="0 Point")
            axes[row, 0].set(xlabel="Data Points", ylabel="Residuals")
            axes[row, 0].legend(loc="upper right")

            axes[row, 1].scatter(truths[pred*l:succ*l], outputs[pred*l:succ*l], c="green")
            axes[row, 1].plot(truths[pred*l:succ*l], truths[pred*l:succ*l], label="Accuracy Line")
            axes[row, 1].set(xlabel="Actual Xe", ylabel="Predicted Xe")
            axes[row, 1].legend(loc="upper right")

            plt.ylim([0, 1])
            
            pred += 1
            succ += 1

        print('Test Loss: {} | Test Baseline: {}\n'.format(
            np.round(test_loss, 3), 
            np.round(self.testbase, 3)))

        axes[0, 0].set_title("Residual plots of predicted and actual eutectic proportion Xe")
        axes[0, 1].set_title("Scatter plots of predicted vs actual eutectic proportion Xe")
        plt.show()

        # fig.savefig('D:\\Research\\UConn_ML\\Images\\snn_results_plots.png')

        # Print the values from the last batch processed just for the user to see
        disp = pd.DataFrame({
            'f(A,B)': np.round(outputs[0:l], 3),
            'f(B,A)': np.round(invouts[0:l], 3),
            'f(A,B) + f(B,A)': outputs[0:l] + invouts[0:l],
            'Truth': np.round(truths[0:l], 3),
            'Pred': np.round(outputs[0:l], 3)})
        
        disp.style.set_properties(**{'text-align': 'center'})
        
        print(disp)