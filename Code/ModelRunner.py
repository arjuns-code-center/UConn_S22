import numpy as np
import torch
from torch.utils.data import DataLoader
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

    def train_and_validate(self):
        tolerance = 0                                # for early stopping
        patience = 0                                 # for callbacks
        
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
                
                patience = 0
            else:
                if patience <= 3:
                    print("Callback to epoch {} | Patience {}/4".format(lowest_loss_epoch, patience+1))
                    
                    checkpoint = torch.load("D:\\Research\\UConn_ML\\Code\\Checkpoints\\checkpoint.pth")
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    patience = patience + 1
                
            # Early stopping. If the loss difference is over some min delta tolerance times, stop as it is not getting better
            # Also stop is train running loss is getting too small and breaks a threshold set
            # Also if the validation loss is over the baseline by a lot, model won't learn anything as we are stuck in a local minima
            if (val_running_loss - train_running_loss) > 0.01:
                tolerance = tolerance + 1
                if tolerance == 5:
                    print("Early Stop. Loss difference over threshold")
                    break
            if train_running_loss <= 0.1:
                print("Early Stop. Train Loss under threshold.")
                break
            if (val_running_loss - self.valbase) > 0.025:
                print("Early Stop. Validation Loss over baseline.")
                break
            
            trloss = np.append(trloss, train_running_loss)
            trbase = np.append(trbase, self.trainbase)
            vloss = np.append(vloss, val_running_loss)
            vbase = np.append(vbase, self.valbase)
            
        return trloss, trbase, vloss, vbase

    def test_plot_stats(self):
        outputs = np.array([]).astype(float)
        invouts = np.array([]).astype(float)
        truths = np.array([]).astype(float)
        
        test_loss = 0.0
        self.model.eval()

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

        print('Test Loss: {} | Test Baseline: {}\n'.format(
            np.round(test_loss, 3), 
            np.round(self.testbase, 3)))

        return outputs, invouts, truths