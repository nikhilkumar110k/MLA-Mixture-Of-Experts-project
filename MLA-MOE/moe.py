import torch
import torch.nn as nn
from experts import Expert 

class MOE(nn.Module):
    def __init__(self,embed,n_experts=4,k=3):
        super().__init__()
        self.k=k
        self.n_experts=n_experts
        self.router=nn.Linear(embed,n_experts)
        self.experts=nn.ModuleList([
            Expert(embed) for _ in range(n_experts)
        ])
    def forward(self,x):
        B,T,C= x.shape

        logits=self.router(x)
        probs= torch.softmax(logits,dim=-1)

        topk_vals,topk_idx=torch.topk(probs,self.k,dim=-1)
        x_flat=x.view(-1,C)
        output=torch.zeros_like(x_flat)
        
        topk_idx=topk_idx.view(-1,self.k)
        topk_vals=topk_vals.view(-1,self.k)

        for expert_id in range(self.n_experts):
            mask= (topk_idx==expert_id)

            if mask.any():
                indices=mask.nonzero(as_tuple=True)[0]
                tokens=x_flat[indices]

                out=self.experts[expert_id](tokens)

                weights=topk_vals.view(-1)[indices].unsqueeze(-1)
                output[indices] +=weights*out 

        importance =probs.mean(dim=(0,1))   
        load = torch.zeros_like(importance)

        for i in range(self.n_experts):
            load[i]= (topk_idx == i).float().mean()

        aux_loss= (importance * load).sum()

        return output.view(B, T, C), aux_loss
                

