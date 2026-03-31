import torch 
import torch.nn as nn 

class Expert(nn.Module):
    def __init__(self,embed):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(embed,embed*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed*4,embed)
        )
    def forward(self,x):
            return self.net(x)