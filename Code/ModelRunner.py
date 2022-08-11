import numpy as np
import torch
from torch.utils.data import DataLoader
import SiameseNeuralNetwork as SNN
import SimpleNeuralNetwork as SimpNN
from sklearn.metrics import r2_score

class MR():
    def __init__(self, starting_features, batchsize, epochs, lr_base, train_dset, val_dset, test_dset, train_median, tp):        
        self.train_ec_dl = DataLoader(train_dset, shuffle=True, batch_size=batchsize, drop_last=True)
        self.val_dset = val_dset
        self.test_dset = test_dset
        
        self.trainlen = len(train_dset)
        self.vallen = len(val_dset)
        self.testlen = len(test_dset)
        
        self.max_epochs = epochs
        self.criterion = torch.nn.L1Loss()
        self.starting_features = starting_features
        self.model = None
        self.lr = lr_base
        
        # Set the baseline calculation parameter, and calculate the baselines
        self.median = train_median
        self.trainbase = 0
        self.valbase = 0
        self.testbase = 0
        
        if tp == "xe":
            for i in train_dset:
                self.trainbase += self.criterion(self.median, i[2]).item() / len(train_dset)
            for i in val_dset:
                self.valbase += self.criterion(self.median, i[2]).item() / len(val_dset)
            for i in test_dset:
                self.testbase += self.criterion(self.median, i[2]).item() / len(test_dset)
        else:
            for i in train_dset:
                self.trainbase += self.criterion(self.median, i[3].float()).item() / len(train_dset)
            for i in val_dset:
                self.valbase += self.criterion(self.median, i[3].float()).item() / len(val_dset)
            for i in test_dset:
                self.testbase += self.criterion(self.median, i[3].float()).item() / len(test_dset)

        # Set the training parameter. Xe or Te
        self.tp = tp

    def train_and_validate(self, modeltype, oB=0):
        self.model = None
        if modeltype == 'siam':
            self.model = SNN.SNN(self.starting_features)
        elif modeltype == 'simp':
            self.model = SimpNN.SimpNN(2*self.starting_features)
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
        patience = 0                                 # for callbacks
        tolerance = 0
        
        trloss = np.array([])
        trbase = np.array([])
        vloss = np.array([])
        vbase = np.array([])
        
        lowest_loss = 100
        lowest_loss_epoch = 0
     
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

            train_running_loss = train_running_loss / x

            self.model.eval()
            for i in range(self.vallen):
                line = self.val_dset[i]

                m1 = line[0].view(1, self.starting_features)
                m2 = line[1].view(1, self.starting_features)

                if self.tp == "xe":
                    truth = line[2].view(1, 1)
                else:
                    truth = line[3].float().view(1, 1)

                output = self.model(m1.float(), m2.float(), self.tp)                
                val_running_loss += self.criterion(output, truth).item()
                
            val_running_loss = val_running_loss / self.vallen
            
            print('Epoch {} | Train Loss: {} | Train Baseline: {} | Val Loss: {} | Val Baseline: {}'.format(
                epoch+1, 
                np.round(train_running_loss, 6), 
                np.round(self.trainbase, 6), 
                np.round(val_running_loss, 6), 
                np.round(self.valbase, 6)))
            
            trloss = np.append(trloss, train_running_loss)
            trbase = np.append(trbase, self.trainbase)
            vloss = np.append(vloss, val_running_loss)
            vbase = np.append(vbase, self.valbase)
            
            # Callback and Early Stopping. 
            # Have criteria to maintain: val loss has to decrease. If constant or increasing, stop. 
            if val_running_loss > oB: # as long as we are over overfit threshold
                if val_running_loss <= lowest_loss: # if loss and rate of loss are ok, then save good model
                    lowest_loss = val_running_loss
                    lowest_loss_epoch = epoch

                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict()}, "D:\\Research\\UConn_ML\\Code\\Checkpoints\\checkpoint.pth")

                    patience = 0
                    tolerance = 0
                elif val_running_loss > oB: # if not ok, find out which one is not ok, and revert model. 
                    checkpoint = torch.load("D:\\Research\\UConn_ML\\Code\\Checkpoints\\checkpoint.pth")
                    
                    # NOTE: lowest_loss_epoch being good could mean good_rate_epoch is bad and vice versa. 
                    if patience < 5:
                        print("Callback to epoch {} | Patience {}/5".format(lowest_loss_epoch+1, patience+1))
                        patience = patience + 1
                    else: # Early stop if after n callbacks the loss has still not gone down
                        print("Early Stop. Validation loss stopped decreasing.")
                        break

                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    tolerance = tolerance + 1
                    
                    if tolerance == 5:
                        print("Early Stop. Validation loss stopped changing.")
                        break
            else:
                print("Early Stop. Validation loss under overfitting threshold.")                
                break
                
        return trloss, trbase, vloss, vbase

    def test(self):
        outputs = np.array([]).astype(float)
        invouts = np.array([]).astype(float)
        truths = np.array([]).astype(float)
        
        test_loss = 0.0
        self.model.eval()
        
        for i in range(self.testlen):
            line = self.test_dset[i]
            
            m1 = line[0].view(1, self.starting_features)
            m2 = line[1].view(1, self.starting_features)
                
            if self.tp == "xe":
                truth = line[2]
            else:
                truth = line[3].float()
            
            truths = np.append(truths, truth[np.newaxis].numpy().T)

            output = self.model(m1.float(), m2.float(), self.tp) # f(A,B)
            invout = self.model(m2.float(), m1.float(), self.tp) # f(B,A)
                
            outputs = np.append(outputs, output.detach().numpy())
            invouts = np.append(invouts, invout.detach().numpy())

            test_loss += self.criterion(output, truth).item()
        
        test_loss = test_loss / self.testlen
        test_r2 = r2_score(truths, outputs)
        
        print('Test Loss: {} | Test Baseline: {} | R^2: {}\n'.format(
            np.round(test_loss, 3), 
            np.round(self.testbase, 3),
            np.round(test_r2, 3)))

        return outputs, invouts, truths, test_loss, test_r2