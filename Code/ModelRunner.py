import numpy as np
import torch
from torch.utils.data import DataLoader
import SiameseNeuralNetwork as SNN
import SimpleNeuralNetwork as SimpNN

class MR():
    def __init__(self, starting_features, batchsize, epochs, lr_base, lr_max, train_dset, val_dset, test_dset, train_stdev, tp):        
        self.train_ec_dl = DataLoader(train_dset, shuffle=True, batch_size=batchsize, drop_last=True)
        self.val_ec_dl = DataLoader(val_dset, shuffle=False, batch_size=len(val_dset))
        self.test_ec_dl = DataLoader(test_dset, shuffle=False, batch_size=len(test_dset))
        
        self.trainlen = len(train_dset)
        self.vallen = len(val_dset)
        self.testlen = len(test_dset)
        
        # Set the model and training parameters
        self.model = SNN.SNN(starting_features)
        self.simpmodel = SimpNN.SimpNN(2*starting_features)
        self.max_epochs = epochs

        # Set optimizer and loss function. Using MAE for regression. CyclicLR scheduler.
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr_base)
        self.sch = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=lr_base, max_lr=lr_max, mode="exp_range")
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

    def train_and_validate(self, oB=0):
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
                self.sch.step()

            train_running_loss = train_running_loss / x

            self.model.eval()
            for v, (m1, m2, xe, Te) in enumerate(self.val_ec_dl):
                if self.tp == "xe":
                    truth = xe
                else:
                    truth = Te.float()

                output = self.model(m1.float(), m2.float(), self.tp)
                val_running_loss += self.criterion(output[:, 0], truth).item()
            
            # val_running_loss = val_running_loss / self.vallen
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
                    if patience < 10:
                        print("Callback to epoch {} | Patience {}/10".format(lowest_loss_epoch+1, patience+1))
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

        with torch.no_grad():
            for t, (m1, m2, xe, Te) in enumerate(self.test_ec_dl):
                if self.tp == "xe":
                    truth = xe
                else:
                    truth = Te.float()
                
                truths = np.append(truths, truth[np.newaxis].numpy().T)

                output = self.model(m1.float(), m2.float(), self.tp) # f(A,B)
                invout = self.model(m2.float(), m1.float(), self.tp) # f(B,A)
                
                outputs = np.append(outputs, output.detach().numpy())
                invouts = np.append(invouts, invout.detach().numpy())

                test_loss += self.criterion(output[:, 0], truth).item()
        
        # test_loss = test_loss / self.testlen
        
        print('Test Loss: {} | Test Baseline: {}\n'.format(
            np.round(test_loss, 3), 
            np.round(self.testbase, 3)))

        return outputs, invouts, truths
    
    def simp_train_and_validate(self, oB=0):
        patience = 0                                 # for callbacks
        tolerance = 0
        
        trloss = np.array([])
        trbase = np.array([])
        vloss = np.array([])
        vbase = np.array([])
        
        lowest_loss = 100
        lowest_loss_epoch = 0
     
        for epoch in range(self.max_epochs):
            self.simpmodel.train()
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
                output = self.simpmodel(m1.float(), m2.float(), self.tp)

                loss = self.criterion(output[:, 0], truth)
                train_running_loss += loss.item()
                
                loss.backward()
                self.opt.step()
                self.sch.step()

            train_running_loss = train_running_loss / x

            self.simpmodel.eval()
            for v, (m1, m2, xe, Te) in enumerate(self.val_ec_dl):
                if self.tp == "xe":
                    truth = xe
                else:
                    truth = Te.float()

                output = self.simpmodel(m1.float(), m2.float(), self.tp)
                val_running_loss += self.criterion(output[:, 0], truth).item()
            
            # val_running_loss = val_running_loss / self.vallen
            
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
                if val_running_loss < lowest_loss: # if loss and rate of loss are ok, then save good model
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

    def simp_test(self):
        outputs = np.array([]).astype(float)
        invouts = np.array([]).astype(float)
        truths = np.array([]).astype(float)
        
        test_loss = 0.0
        self.simpmodel.eval()

        with torch.no_grad():
            for t, (m1, m2, xe, Te) in enumerate(self.test_ec_dl):
                if self.tp == "xe":
                    truth = xe
                else:
                    truth = Te.float()
                
                truths = np.append(truths, truth[np.newaxis].numpy().T)

                output = self.simpmodel(m1.float(), m2.float(), self.tp) # f(A,B)
                invout = self.simpmodel(m2.float(), m1.float(), self.tp) # f(B,A)
                
                outputs = np.append(outputs, output.detach().numpy())
                invouts = np.append(invouts, invout.detach().numpy())

                test_loss += self.criterion(output[:, 0], truth).item()
        
        # test_loss = test_loss / self.testlen

        print('Test Loss: {} | Test Baseline: {}\n'.format(
            np.round(test_loss, 3), 
            np.round(self.testbase, 3)))

        return outputs, invouts, truths