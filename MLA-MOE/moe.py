import torch
import torch.nn as nn
from experts import Expert 

class MOE(nn.Module):
    def __init__(self,embed,n_experts=4,k=3):
        super().__init__()
        self.k=k
        self.n_experts=n_experts
        self.router=nn.Linear(embed,n_experts)
        self.expert=nn.ModuleList([
            Expert(embed) for _ in range(n_experts)
        ])