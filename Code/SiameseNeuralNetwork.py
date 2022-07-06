import torch
from torch.nn import Module, Linear, Sequential, Softplus, Tanh, Sigmoid

# SNN class with model
class SNN(Module):
    def __init__(self, start_features): 
        super(SNN, self).__init__()
        
        self.model = Sequential(
            Linear(in_features=start_features, out_features=4, bias=False),
            Linear(in_features=4, out_features=3, bias=False),
            Softplus()
        )
        
        torch.nn.kaiming_uniform_(self.model.parameters(), nonlinearity='softplus')
        
        self.output = Sequential(
            Linear(in_features=6, out_features=3, bias=False), # distance metric calculation stage
            Tanh(),
            Linear(in_features=3, out_features=1, bias=False), # final stage where prediction is made
            Sigmoid()
        )
        
        torch.nn.xavier_uniform_(self.output.parameters())

    def forward(self, x1, x2, marker="xe"):
        if marker == "xe":
            return self.xe(x1, x2)
        elif marker == "Te":
            return self.te(x1, x2)
        
    def xe(self, x1, x2):
        y1 = self.model(x1)               # (batchsize, 3)
        y2 = self.model(x2)               # (batchsize, 3)
        y = torch.cat([y1, y2], 1)        # (batchsize, 6)
        
        # for property f(A,B) = 1 - f(B,A)
        with torch.no_grad():
            self.output[0].weight[:, 3:6] = -1 * self.output[0].weight[:, 0:3]
            
            self.output[0].weight[0, 1:2] = 0
            self.output[0].weight[0, 4:5] = 0
            
            self.output[0].weight[1, 0] = 0
            self.output[0].weight[1, 2:3] = 0
            self.output[0].weight[1, 5] = 0
            
            self.output[0].weight[2, 0:1] = 0
            self.output[0].weight[2, 3:4] = 0
        
        p = self.output(y)                # output for xe (batchsize, 1)
        return p
    
    def te(self, x1, x2):
        y1 = self.model(x1)               # (batchsize, 3)
        y2 = self.model(x2)               # (batchsize, 3)
        y = torch.cat([y1, y2], 1)        # (batchsize, 6)
        p = self.output(y)                # (batchsize, 1)
        return p