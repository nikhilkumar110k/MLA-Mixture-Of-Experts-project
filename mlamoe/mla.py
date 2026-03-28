import torch 
import torch.nn as nn 

class MLA(nn.Module):
    def __init__(self,embed_size,num_heads,d_latent=64):
     super().__init__()
     self.embed_size=embed_size
     self.num_heads=num_heads
     self.d_latent=d_latent
     self.head_dim= embed_size//num_heads

     self.q= nn.Linear(embed_size,embed_size)
     self.k= nn.Linear(embed_size,embed_size)
     self.v= nn.Linear(embed_size,embed_size)
     
     self.w_down= nn.Linear(embed_size,d_latent)
     self.w_up=nn.Linear(d_latent,embed_size)

     self.out=nn.Linear(embed_size,embed_size)

    def forward(self,x):
       B,T,C=x.shape
       Q=self.q(x)
       K=self.k(x)
       V=self.v(x)

       Z_k=self.w_down(K)
       Z_v=self.w_down(V)

       K=self.w_up(Z_k)
       V=self.w_up(Z_v)

       Q= Q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
       K= K.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
       V= V.view(B,T,self.num_heads,self.head_dim).transpose(1,2)

       attn= (Q@K.transpose(-2,-1))/(self.head_dim ** 0.5)
       attn= torch.softmax(attn,dim=-1)
       out=attn@V
       out=out.transpose(1,2).contiguous().reshape(B,T,C)

       return self.out(out)