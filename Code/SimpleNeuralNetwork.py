import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

# Simple NN class model
class SimpNN(Module):
    def __init__(self, start_features): 
        super(SimpNN, self).__init__()
        
        self.fc1 = Linear(in_features=start_features, out_features=15, bias=False)
        self.fc2 = Linear(in_features=15, out_features=5, bias=False)
        
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu', mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu', mode='fan_in')
        
        self.p = Linear(in_features=5, out_features=1, bias=False)   # output stage
        
        torch.nn.init.xavier_normal_(self.p.weight)
        
    def forward(self, x1, x2, marker="xe"):        
        if marker == "xe":
            return self.xe(x1, x2)
        elif marker == "Te":
            return self.te(x1, x2)
        
    def xe(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        y = F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x))))      # (batchsize, 5)
        p = torch.sigmoid(self.p(y))                               # (batchsize, 1)
        return p
    
    def te(self, x1, x2):
        x = torch.cat([x1, x2], 1)            
        y = F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x))))      # (batchsize, 5)
        p = torch.sigmoid(self.p(y))                               # (batchsize, 1)
        return p