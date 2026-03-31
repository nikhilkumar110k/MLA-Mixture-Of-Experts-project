🚀 MLA + MoE vs Dense Transformer (T5-style) Benchmark

This project implements and benchmarks a Mixture of Experts (MoE) + Multi-head Latent Attention (MLA) architecture against a dense Transformer (T5-style classifier) on the IMDB sentiment classification task.

---

📌 Overview

We compare two architectures:

- Dense Transformer (T5-style / GPTClassifier)
- MLA + MoE (Sparse Transformer)

The goal is to evaluate:

- Accuracy
- Training efficiency
- Memory usage
- Parameter efficiency
- Stability of training

---

🧠 Architecture

🔹 Dense Model (Baseline)

- Standard Transformer encoder
- Fully dense feed-forward layers
- All parameters active per forward pass

---

🔹 MLA + MoE Model

1. Multi-head Latent Attention (MLA)

- Compresses key/value representations into a latent space
- Reduces attention memory footprint
- Reconstructs attention outputs via projection

2. Mixture of Experts (MoE)

- Multiple feed-forward "experts"
- Router selects top-k experts per token
- Sparse computation (only selected experts activated)

---

⚙️ Key Features

- ✅ Custom Transformer implementation (no HuggingFace dependency)
- ✅ Sparse MoE routing with top-k selection
- ✅ Auxiliary load balancing loss
- ✅ Benchmarking pipeline
- ✅ GPU support (CUDA)

---

📂 Project Structure

MLA-MOE-project/
│
├── main.py                
├── train.py                
├── benchmark.py          # Benchmarking script
│
├── mlamoe/
│   ├── mlamoe.py         # MLA + MoE blocks
│   ├── moe.py            # MoE routing + experts
│   ├── mla.py            # MLA attention
│   ├── experts.py        # Expert FFN
│
├── selfattention/
│   ├── tselfattention.py # Dense Transformer (baseline)
│
├── checkpoints/          # Saved models

---

📊 Dataset

- IMDB Sentiment Dataset
- Binary classification (positive / negative)
- Training subset: 20,000 samples
- Validation subset: 2,000 samples

---

🚀 How to Run

1️⃣ Install dependencies

uv venv
uv pip install torch datasets

---

2️⃣ Run training + benchmarking

uv run main.py

uv run benchmark.py
---

📈 Example Results

Model| Accuracy| Time/Epoch| Memory| Params
Dense (T5)| ~0.88| ~120s| ~1.3GB| 23M
MLA + MoE| ~0.83| ~300–400s| ~3GB| 64M



