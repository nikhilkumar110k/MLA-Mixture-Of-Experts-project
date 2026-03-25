import torch
import torch.nn as nn
from mla import MLA
from moe import MOE


class MLAMOE(nn.Modules):
    def __init__(self,embed,heads,d_latent=64,n_experts=4):
        super().__init__()

        self.ln1=nn.LayerNorm(embed)
        self.attn=MLA(embed,heads,d_latent)

        self.ln2= nn.LayerNorm(embed)
        self.moe=MOE(embed,n_experts=n_experts)

    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.moe(self.ln2(x))
        return x

