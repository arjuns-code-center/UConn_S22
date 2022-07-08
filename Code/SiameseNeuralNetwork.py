import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

# SNN class with model
class SNN(Module):
    def __init__(self, start_features): 
        super(SNN, self).__init__()
        
        self.fc1 = Linear(in_features=start_features, out_features=4, bias=False)
        self.fc2 = Linear(in_features=4, out_features=3, bias=False)
        
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        
        self.dA = Linear(in_features=2, out_features=1, bias=False)
        self.dB = Linear(in_features=2, out_features=1, bias=False)
        self.dC = Linear(in_features=2, out_features=1, bias=False)  # distance metric
        
        torch.nn.init.xavier_uniform_(self.dA.weight)
        torch.nn.init.xavier_uniform_(self.dB.weight)
        torch.nn.init.xavier_uniform_(self.dC.weight)
        
        self.p = Linear(in_features=3, out_features=1, bias=False)   # output stage
        
        torch.nn.init.xavier_uniform_(self.p.weight)
 
    def forward(self, x1, x2, marker="xe"):        
        if marker == "xe":
            return self.xe(x1, x2)
        elif marker == "Te":
            return self.te(x1, x2)
        
    def xe(self, x1, x2):
        y1 = F.relu(self.fc2(F.relu(self.fc1(x1))))             # (batchsize, 3)
        y2 = F.relu(self.fc2(F.relu(self.fc1(x2))))             # (batchsize, 3)
        
        try:
            y = torch.cat([y1, y2], 1)                          # (batchsize, 6)
        except IndexError:
            y1 = y1.view(1, len(y1))
            y2 = y2.view(1, len(y2))
            y = torch.cat([y1, y2], 1)
        
        yA = torch.cat([y[:, 0].view(len(y), 1), y[:, 3].view(len(y), 1)], 1)  # (batchsize, 2) each
        yB = torch.cat([y[:, 1].view(len(y), 1), y[:, 4].view(len(y), 1)], 1)
        yC = torch.cat([y[:, 2].view(len(y), 1), y[:, 5].view(len(y), 1)], 1)
        
        # for property f(A,B) = 1 - f(B,A)
        with torch.no_grad():
            self.dA.weight[:, 1] = -1 * self.dA.weight[:, 0]
            self.dB.weight[:, 1] = -1 * self.dB.weight[:, 0]
            self.dC.weight[:, 1] = -1 * self.dC.weight[:, 0]
        
        zA = torch.tanh(self.dA(yA))                            # (batchsize, 1)
        zB = torch.tanh(self.dB(yB))
        zC = torch.tanh(self.dC(yC))
        
        z = torch.cat([zA, zB, zC], 1)                          # (batchsize, 3)
        
        p = torch.sigmoid(self.p(z))                            # output for xe (batchsize, 1)
        return p
    
    def te(self, x1, x2):
        y1 = F.softplus(self.fc2(self.fc1(x1)))                 # (batchsize, 3)
        y2 = F.softplus(self.fc2(self.fc1(x2)))                 # (batchsize, 3)
        
        try:
            y = torch.cat([y1, y2], 1)                          # (batchsize, 6)
        except IndexError:
            y1 = y1.view(1, len(y1))
            y2 = y2.view(1, len(y2))
            y = torch.cat([y1, y2], 1)
            
        yA = torch.cat([y[:, 0], y[:, 3]], 1)                   # (batchsize, 2) each
        yB = torch.cat([y[:, 1], y[:, 4]], 1)
        yC = torch.cat([y[:, 2], y[:, 5]], 1)
        
        zA = torch.tanh(torch.abs(self.dA(yA)))                 # (batchsize, 1)
        zB = torch.tanh(torch.abs(self.dB(yB)))
        zC = torch.tanh(torch.abs(self.dC(yC)))
        
        z = torch.cat([zA, zB, zC], 1)                          # (batchsize, 3)
            
        p = self.p(z)                                           # (batchsize, 1)
        return p