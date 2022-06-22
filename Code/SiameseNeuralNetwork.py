import torch
from torch.nn import Module, Linear, Sequential, Sigmoid

# SNN class with model
class SNN(Module):
    def __init__(self, start_features): 
        super(SNN, self).__init__()
        
        self.model = Sequential(
            Linear(in_features=start_features, out_features=4, bias=False),
            Linear(in_features=4, out_features=3, bias=False),
        )
        
        self.output = Sequential(
            Linear(in_features=6, out_features=6, bias=False), # distance metric calculation stage
            Linear(in_features=6, out_features=1, bias=False), # final stage where prediction is made
            Sigmoid()
        )

    def forward(self, x1, x2, marker="xe"):
        if marker == "xe":
            return self.xe(x1, x2)
        elif marker == "Te":
            return self.te(x1, x2)
        
    def xe(self, x1, x2):
        y1 = self.model(x1)               # (batchsize, 3)
        y2 = self.model(x2)               # (batchsize, 3)
        y = torch.cat([y1, y2], 1)        # (batchsize, 6)
        
        with torch.no_grad():
            self.output[0].weight[:, 3:6] = -1 * self.output[0].weight[:, 0:3]
        
        p = self.output(y)                # output for xe   (batchsize, 1)
        return p
    
    def te(self, x1, x2):
        y1 = self.model(x1)
        y2 = self.model(x2)   
        d = torch.abs(y1 - y2)           # difference for Te
        p = self.fc(d)                   # output for Te
        return p