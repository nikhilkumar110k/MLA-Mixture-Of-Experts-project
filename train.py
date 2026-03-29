from datasets import load_dataset
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn as nn 
from selfattention.tselfattention import GPTClassifier
from mlamoe.mlamoe import MLAMOEClassifier
from torch.nn.utils import clip_grad_norm_
import os 

device= "cuda" if torch.cuda.is_available() else "cpu"

dataset= load_dataset("imdb")
train_texts=dataset["train"]["text"][:20000]
train_labels=dataset["train"]["label"][:20000]

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


def encode(text,vocab,max_len=128):
    tokens=text.lower().split()
    ids=[vocab.get(w,1) for w in tokens][:max_len]
    ids+=[0]*(max_len -len(ids))
    return torch.tensor(ids)


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

model_t5= GPTClassifier(vocab_size)
model_mlamoe= MLAMOEClassifier(vocab_size)

def train_one_epoch(model,loader,optimizer):
    model.train()
    total_loss=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)

        optimizer.zero_grad()

        out=model(x)

        if isinstance(out,tuple):
            logits,aux_loss=out
            loss=F.cross_entropy(logits,y)+ 0.01 * aux_loss
        else: 
            logits= out
            loss= F.cross_entropy(logits,y)
        loss.backward()
        clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        total_loss+=loss.item()
    return total_loss/len(loader)

def evaluate(model,loader):
    model.eval()
    correct=0
    total=0

    with torch.no_grad():
        for x,y in loader:
            x,y= x.to(device),y.to(device)
            out=model(x)
            logits=out[0] if isinstance(out,tuple) else out
            preds= torch.argmax(logits,dim=-1)

            correct+=(preds==y).sum().item()
            total+=y.size(0)
        return correct/total

def run_model(model,name):
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
    save_path = f"checkpoints/{name}"
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(5):
        start= time.time()

        loss= train_one_epoch(model,train_loader,optimizer)
        acc= evaluate(model,val_loader)
        print(f"model type: {name}")

        print(f"epoch {epoch}")
        print(f"loss {loss}")
        print(f"accuracy {acc}")
        print(f"time_taken: {time.time()-start:.2f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": acc,
                "epoch": epoch
            }, f"{save_path}/best{name}.pt")
    torch.save(model.state_dict(), f"{save_path}/final{name}.pt")
    memory=torch.cuda.max_memory_allocated() / 1e6 if device=="cuda" else 0
    params=sum(p.numel() for p in model.parameters())

    return acc,memory,params
    
