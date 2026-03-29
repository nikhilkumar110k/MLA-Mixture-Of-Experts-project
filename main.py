import torch
import torch.nn as nn
from selfattention.tselfattention import T5Attention
from mlamoe.mlamoe import MLAMOE
from train import run_model,model_t5,model_mlamoe

device= "cuda" if torch.cuda.is_available else "cpu"
acc_t5, mem_t5, params_t5 = run_model(model_t5, "T5")

acc_moe, mem_moe, params_moe = run_model(model_mlamoe, "MLA+MoE")