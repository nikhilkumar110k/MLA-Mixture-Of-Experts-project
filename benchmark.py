import torch
import time
from torch.utils.data import Dataset, DataLoader
from selfattention.tselfattention import GPTClassifier
from mlamoe.mlamoe import MLAMOEClassifier
from datasets import load_dataset
from collections import Counter

dataset= load_dataset("imdb")

train_texts=dataset["train"]["text"][:20000]
train_labels=dataset["train"]["label"][:20000]

def encode(text,vocab,max_len=128):
    tokens=text.lower().split()
    ids=[vocab.get(w,1) for w in tokens][:max_len]
    ids+=[0]*(max_len -len(ids))
    return torch.tensor(ids)

val_texts=dataset["test"]["text"][:2000]
val_labels=dataset["test"]["label"][:2000]



def build_vocab(texts,max_size=20000):
    counter=Counter()

    for text in texts:
        counter.update(text.lower().split())

    vocab={"<pad>":0,"<unk>":1}
    for i,(word,_) in enumerate(counter.most_common(max_size-2)):
        vocab[word]=i+2
    return vocab


vocab=build_vocab(train_texts)
vocab_size=len(vocab)
class IMDBdataset(Dataset):
    def __init__(self,texts,labels,vocab):
        self.texts=texts
        self.labels=labels
        self.vocab=vocab
    def __len__(self):
        return len(self.texts)
    def __getitem__(self,idx):
        x=encode(self.texts[idx],self.vocab)
        y= torch.tensor(self.labels[idx])
        return x,y

train_dataset=IMDBdataset(train_texts,train_labels,vocab)
val_dataset=IMDBdataset(val_texts,val_labels,vocab)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset= load_dataset("imdb")
train_texts=dataset["train"]["text"][:20000]
train_labels=dataset["train"]["label"][:20000]

val_texts=dataset["test"]["text"][:2000]
val_labels=dataset["test"]["label"][:2000]





def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def measure_latency(model, loader, steps=50):
    model.eval()
    latencies = []
    total_tokens = 0

    data_iter = iter(loader)

    for _ in range(10):
        x, _ = next(data_iter)
        x = x.to(device)
        _ = model(x)

    torch.cuda.synchronize()

    for _ in range(steps):
        x, _ = next(data_iter)
        x = x.to(device)

        start = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        end = time.time()

        latencies.append(end - start)
        total_tokens += x.numel()

    avg_latency = sum(latencies) / len(latencies)
    throughput = total_tokens / sum(latencies)

    return avg_latency * 1000, throughput


def get_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark(model, loader, name):
    print(f"\n===== {name} =====")

    model.to(device)
    torch.cuda.reset_peak_memory_stats()

    acc = evaluate(model, loader)
    latency, throughput = measure_latency(model, loader)
    memory = get_memory()
    params = count_params(model)

    print(f"Accuracy: {acc:.4f}")
    print(f"Latency (ms): {latency:.2f}")
    print(f"Throughput: {throughput:.2f}")
    print(f"Memory (MB): {memory:.2f}")
    print(f"Params: {params}")

    return acc, latency, memory




model_t5 = GPTClassifier(vocab_size)
model_moe = MLAMOEClassifier(vocab_size)

ckpt1 = torch.load("checkpoints/T5/bestT5.pt", map_location=device)
model_t5.load_state_dict(ckpt1["model_state_dict"])
ckpt2 = torch.load("checkpoints/MLA+MoE/bestMLA+MOE.pt", map_location=device)
model_moe.load_state_dict(ckpt2["model_state_dict"])

res_t5 = benchmark(model_t5, val_loader, "T5")
res_moe = benchmark(model_moe, val_loader, "MLA+MoE")

print("\n===== WINNER =====")
print("Accuracy:", "MLA+MoE" if res_moe[0] > res_t5[0] else "T5")
print("Speed:", "MLA+MoE" if res_moe[1] < res_t5[1] else "T5")
print("Memory:", "MLA+MoE" if res_moe[2] < res_t5[2] else "T5")