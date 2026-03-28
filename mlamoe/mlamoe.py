import torch
import torch.nn as nn
from mla import MLA
from moe import MOE



class MLAMOE(nn.Module):
    def __init__(self, embed, heads, d_latent=64, n_experts=4):
        super().__init__()

        self.ln1= nn.LayerNorm(embed)
        self.attn=MLA(embed, heads, d_latent)

        self.ln2= nn.LayerNorm(embed)

        self.shared=nn.Sequential(
            nn.Linear(embed, embed * 4),
            nn.GELU(),
            nn.Linear(embed * 4, embed)
        )

        self.moe = MOE(embed, n_experts=n_experts)

    def forward(self, x):
        x=x + self.attn(self.ln1(x))

        h =self.ln2(x)

        shared_out = self.shared(h)
        moe_out, aux_loss = self.moe(h)

        x = x + shared_out + moe_out

        return x, aux_loss

class MLAMOEClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=400, context=312, n_heads=8, n_layers=8):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.positions = nn.Embedding(context, embed_size)

        self.blocks = nn.ModuleList([
            MLAMOE(embed_size, n_heads)  
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(embed_size)

        self.classifier = nn.Linear(embed_size, 2)

    def forward(self, inp):
        B, L = inp.size()

        tok=self.embeddings(inp)

        pos=self.positions(torch.arange(L, device=inp.device))
        pos=pos.unsqueeze(0).expand(B, L, -1)

        x =tok+pos

        total_aux_loss =None

        for block in self.blocks:
                x, aux_loss = block(x)
                total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss

        x = self.ln(x)

        pooled = x.mean(dim=1)

        logits = self.classifier(pooled)

        return logits, total_aux_loss