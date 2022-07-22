import numpy as np
import torch
from torch.utils.data import DataLoader
import SiameseNeuralNetwork as SNN

class MR():
    def __init__(self, starting_features, batchsize, epochs, lr_base, lr_max, train_dset, val_dset, test_dset, train_stdev, tp):        
        self.train_ec_dl = DataLoader(train_dset, shuffle=True, batch_size=batchsize, drop_last=True)
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.test_dset = test_dset
        
        # Set the model and training parameters
        self.model = SNN.SNN(starting_features)
        self.max_epochs = epochs

        # Set optimizer and loss function. Using MAE for regression. CyclicLR scheduler.
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr_base)
        self.sch = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=lr_base, max_lr=lr_max, mode="triangular", cycle_momentum=False)
        self.criterion = torch.nn.L1Loss()
        
        # Set the baseline calculation parameter, and calculate the baselines
        self.stdev = train_stdev
        self.trainbase = 0
        self.valbase = 0
        self.testbase = 0
        
        if tp == "xe":
            for i in train_dset:
                self.trainbase += self.criterion(self.stdev, i[2]).item() / len(train_dset)
            for i in val_dset:
                self.valbase += self.criterion(self.stdev, i[2]).item() / len(val_dset)
            for i in test_dset:
                self.testbase += self.criterion(self.stdev, i[2]).item() / len(test_dset)
        else:
            for i in train_dset:
                self.trainbase += self.criterion(self.stdev, i[3].float()).item() / len(train_dset)
            for i in val_dset:
                self.valbase += self.criterion(self.stdev, i[3].float()).item() / len(val_dset)
            for i in test_dset:
                self.testbase += self.criterion(self.stdev, i[3].float()).item() / len(test_dset)

        # Set the training parameter. Xe or Te
        self.tp = tp

    def train_and_validate(self, dB=1, oB=0):
        tolerance = 0                                # for early stopping
        patience = 0                                 # for callbacks
        
        trloss = np.zeros((self.max_epochs+1))
        trbase = np.zeros((self.max_epochs+1))
        vloss = np.zeros((self.max_epochs+1))
        vbase = np.zeros((self.max_epochs+1))
        
        lowest_loss = 100
        good_loss = 100
        lowest_loss_epoch = 0
        good_rate_epoch = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_running_loss = 0.0
            val_running_loss = 0.0
            
            # training step: iterate through the batch and obtain the 4 data
            for x, (m1, m2, xe, Te) in enumerate(self.train_ec_dl):     
                self.opt.zero_grad()
                
                if self.tp == "xe":
                    truth = xe
                else:
                    truth = Te.float()

                # pass 2 sets of inputs into the snn and get p, the output
                output = self.model(m1.float(), m2.float(), self.tp)

                loss = self.criterion(output[:, 0], truth)
                train_running_loss += loss.item()
                
                loss.backward()
                self.opt.step()
                self.sch.step()

            train_running_loss = train_running_loss / x

            self.model.eval()
            for line in self.val_dset:
                m1 = line[0]
                m2 = line[1]
                
                if self.tp == "xe":
                    truth = line[2]
                else:
                    truth = line[3].float()
                truth = torch.unsqueeze(truth, 0)

                output = self.model(m1.float(), m2.float(), self.tp)
                val_running_loss += self.criterion(output[:, 0], truth).item()
            
            val_running_loss = val_running_loss / len(self.val_dset)
            print('Epoch {} | Train Loss: {} | Train Baseline: {} | Val Loss: {} | Val Baseline: {}'.format(
                epoch+1, 
                np.round(train_running_loss, 6), 
                np.round(self.trainbase, 6), 
                np.round(val_running_loss, 6), 
                np.round(self.valbase, 6)))
            
            trloss[epoch] = train_running_loss
            trbase[epoch] = self.trainbase
            vloss[epoch] = val_running_loss
            vbase[epoch] = self.valbase
            
            # Callback and Early Stopping. 
            # Have 2 criteria to maintain: 1. val loss has to decrease. 2. rate of val loss w.r.t epochs (dvde) should not be too high.
            # Update model and optimizer anyways. But since scheduler controls lr which affects derivative, update iff derivative too high. 
            if val_running_loss > oB and epoch != 0: # as long as we are over overfit threshold
                dvde = vloss[good_rate_epoch] - vloss[epoch]
                
                if val_running_loss <= lowest_loss and dvde < dB: # if loss and rate of loss are ok, then save good model
                    lowest_loss = val_running_loss
                    lowest_loss_epoch = epoch
                    good_loss = val_running_loss
                    good_rate_epoch = epoch

                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(), 
                        'scheduler_state_dict': self.sch.state_dict()}, "D:\\Research\\UConn_ML\\Code\\Checkpoints\\checkpoint.pth")

                    patience = 0
                    tolerance = 0
                else: # if not ok, find out which one is not ok, and revert model. 
                    checkpoint = torch.load("D:\\Research\\UConn_ML\\Code\\Checkpoints\\checkpoint.pth")
                    
                    # NOTE: lowest_loss_epoch being good could mean good_rate_epoch is bad and vice versa. 
                    if val_running_loss > lowest_loss:
                        if patience < 5:
                            print("Callback to epoch {} | Patience {}/5".format(lowest_loss_epoch+1, patience+1))
                            patience = patience + 1
                        else: # Early stop if after n callbacks the loss has still not gone down
                            print("Early Stop. Validation loss stopped decreasing.")
                            break
                    elif dvde > dB:
                        if tolerance < 5:
                            print("Callback to epoch {} | Tolerance {}/5".format(good_rate_epoch+1, tolerance+1))
                            tolerance = tolerance + 1
                            self.sch.load_state_dict(checkpoint['scheduler_state_dict'])

                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            elif val_running_loss < oB:
                print("Early Stop. Validation loss under overfitting threshold.")                
                break
            
        trloss = np.trim_zeros(trloss, 'b')
        trbase = np.trim_zeros(trbase, 'b')
        vloss = np.trim_zeros(vloss, 'b')
        vbase = np.trim_zeros(vbase, 'b')
                
        return trloss, trbase, vloss, vbase

    def test(self):
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
                    truth = line[2]
                else:
                    truth = line[3].float()
                
                truths = np.append(truths, truth[np.newaxis].numpy().T)
                truth = torch.unsqueeze(truth, 0)

                output = self.model(m1.float(), m2.float(), self.tp) # f(A,B)
                invout = self.model(m2.float(), m1.float(), self.tp) # f(B,A)
                
                outputs = np.append(outputs, output.detach().numpy())
                invouts = np.append(invouts, invout.detach().numpy())

                test_loss += self.criterion(output[:, 0], truth).item() / len(self.test_dset)

        print('Test Loss: {} | Test Baseline: {}\n'.format(
            np.round(test_loss, 3), 
            np.round(self.testbase, 3)))

        return outputs, invouts, truths