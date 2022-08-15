import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

# Siamese NN class model
class SNN(Module):
    def __init__(self, start_features): 
        super(SNN, self).__init__()
        
        self.fc1 = Linear(in_features=start_features, out_features=10, bias=False)
        self.fc2 = Linear(in_features=10, out_features=7, bias=False)
        self.fc3 = Linear(in_features=7, out_features=5, bias=False)
        self.fc4 = Linear(in_features=5, out_features=3, bias=False)
        
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu', mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu', mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu', mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu', mode='fan_in')
        
        self.dA = Linear(in_features=2, out_features=1, bias=False)
        self.dB = Linear(in_features=2, out_features=1, bias=False)
        self.dC = Linear(in_features=2, out_features=1, bias=False)  # distance metric
        
        torch.nn.init.xavier_normal_(self.dA.weight)
        torch.nn.init.xavier_normal_(self.dB.weight)
        torch.nn.init.xavier_normal_(self.dC.weight)
        
        self.p = Linear(in_features=3, out_features=1, bias=False)   # output stage
        
        torch.nn.init.xavier_normal_(self.p.weight)
 
    def forward(self, x1, x2, marker="xe"):        
        if marker == "xe":
            return self.xe(x1, x2)
        elif marker == "Te":
            return self.te(x1, x2)
        
    def xe(self, x1, x2):
        y1 = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x1))))))))      # (batchsize, 3)
        y2 = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x2))))))))      # (batchsize, 3)        
        y = torch.cat([y1, y2], 1)
        
        yA = torch.cat([y[:, 0].view(len(y), 1), y[:, 3].view(len(y), 1)], 1)  # (batchsize, 2) each
        yB = torch.cat([y[:, 1].view(len(y), 1), y[:, 4].view(len(y), 1)], 1)
        yC = torch.cat([y[:, 2].view(len(y), 1), y[:, 5].view(len(y), 1)], 1)
        
        # for property f(A,B) = 1 - f(B,A)
        with torch.no_grad():
            self.dA.weight[:, 1] = -1 * self.dA.weight[:, 0]
            self.dB.weight[:, 1] = -1 * self.dB.weight[:, 0]
            self.dC.weight[:, 1] = -1 * self.dC.weight[:, 0]
        
        # weighted difference layer
        zA = self.dA(yA)                                              # (batchsize, 1)
        zB = self.dB(yB)
        zC = self.dC(yC)
        
        z = torch.cat([zA, zB, zC], 1)                                # (batchsize, 3)
        
        p = torch.sigmoid(self.p(z))                      # output for xe (batchsize, 1)
        return p
    
    def te(self, x1, x2):
        y1 = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x1))))))))      # (batchsize, 3)
        y2 = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x2))))))))      # (batchsize, 3)
        y = torch.cat([y1, y2], 1)
            
        yA = torch.cat([y[:, 0].view(len(y), 1), y[:, 3].view(len(y), 1)], 1)  # (batchsize, 2) each
        yB = torch.cat([y[:, 1].view(len(y), 1), y[:, 4].view(len(y), 1)], 1)
        yC = torch.cat([y[:, 2].view(len(y), 1), y[:, 5].view(len(y), 1)], 1)
        
        # weighted difference
        with torch.no_grad():
            self.dA.weight[:, 1] = -1 * self.dA.weight[:, 0]
            self.dB.weight[:, 1] = -1 * self.dB.weight[:, 0]
            self.dC.weight[:, 1] = -1 * self.dC.weight[:, 0]
        
        zA = torch.abs(self.dA(yA))                                  # (batchsize, 1)
        zB = torch.abs(self.dB(yB))
        zC = torch.abs(self.dC(yC))
        
        z = torch.cat([zA, zB, zC], 1)                               # (batchsize, 3)
            
        p = self.p(z)                                 # output for Te (batchsize, 1)
        return p