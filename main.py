import torch
import torch.nn as nn
from selfattention.tselfattention import T5Attention
from mlamoe.mlamoe import MLAMOE
from train import run_model,model_t5,model_mlamoe

device= "cuda" if torch.cuda.is_available() else "cpu"
#torch.cuda.reset_peak_memory_stats()
# print("1st time memory cleared")
# acc_t5, mem_t5, params_t5 = run_model(model_t5, "T5")
# print(f"accuracy of t5 {acc_t5}")
# print(f"memory taken by t5 {mem_t5}")
# print(f"total parameters of t5 model {params_t5}")



torch.cuda.reset_peak_memory_stats()
print("second time memory cleared")
acc_moe, mem_moe, params_moe = run_model(model_mlamoe, "MLA+MoE")
print(f"accuracy of moe {acc_moe}")
print(f"memory taken by moe {mem_moe}")
print(f"total parameters of moe model {params_moe}")

