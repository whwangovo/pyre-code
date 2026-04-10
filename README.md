[English](./README.md) | [中文](./README_CN.md)

<p align="center">
  <h1 align="center">🔥 Pyre Code</h1>
  <p align="center">
    Implement the internals of modern AI systems from scratch — Transformers, vLLM, TRL, and beyond.
  </p>
  <p align="center">
    <em>Read the paper, then write the code. No GPU required.</em>
  </p>
  <p align="center">
    <a href="https://star-history.com/#whwangovo/pyre-code&Date">
      <img src="https://img.shields.io/github/stars/whwangovo/pyre-code?style=social" alt="GitHub stars" />
    </a>
  </p>
</p>

---

## 🧠 What is Pyre Code?

68 problems. You write the implementation, a local grading service runs the tests, you see what broke. That's it.

The problems cover what's actually inside Transformers, vLLM, TRL, and diffusion models — attention variants, training tricks, inference kernels, alignment algorithms. No GPU needed.

### Who is this for?

- **Preparing for ML interviews** — practice implementing core components under test, not just reading about them
- **Learning by building** — if you learn best by writing code rather than watching lectures, this is your gym
- **Deepening your understanding** — you've used `nn.MultiheadAttention`, now write it yourself

### Features

- **Browser editor** — Monaco with Python syntax highlighting, no IDE setup
- **Instant feedback** — submit and see pass/fail per test case in seconds
- **Reference solutions** — compare after your own attempt
- **Progress tracking** — solved count and attempt history, persisted across sessions
- **Fully local** — nothing leaves your machine

### Tech Stack

| Layer        | Technology                                                                           |
| ------------ | ------------------------------------------------------------------------------------ |
| Frontend     | Next.js + Monaco Editor + Tailwind CSS                                               |
| Backend      | FastAPI grading service                                                              |
| Judge Engine | [torch_judge](https://github.com/duoan/TorchCode) — executes and validates submissions |
| Storage      | SQLite (progress tracking)                                                           |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+

### Installation

**Option A — one-liner (recommended)**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
./setup.sh
npm run dev
```

`setup.sh` automatically creates a `.venv` Python environment (prefers `uv`, falls back to `python3 -m venv`), installs all dependencies, then prints the start command.

**Option B — conda**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
conda create -n pyre python=3.11 -y && conda activate pyre
pip install -e ".[dev]"
npm install
npm run dev   # run with conda env activated
```

**Option C — manual (venv)**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code

# create a Python env — pick one:
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

npm install
npm run dev
```

Either way, once running:

- **Grading service** → `http://localhost:8000`
- **Web app** → `http://localhost:3000`

**Option D — Docker**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
docker compose up --build
```

Open `http://localhost:3000`. Progress is persisted in a Docker volume. Run `docker compose down -v` to reset.

---

## 📋 Problem Set

68 problems organized by category:

| Category | Problems |
|---|---|
| **Fundamentals** | ReLU, Softmax, GELU, SwiGLU, Dropout, Embedding, Linear, Kaiming Init, Linear Regression |
| **Normalization** | LayerNorm, BatchNorm, RMSNorm |
| **Attention** | Scaled Dot-Product, Multi-Head, Causal, Cross, GQA, Sliding Window, Linear, Flash, Differential, MLA |
| **Position Encoding** | Sinusoidal PE, RoPE, ALiBi, NTK-aware RoPE |
| **Architecture** | SwiGLU MLP, GPT-2 Block, ViT Patch, ViT Block, Conv2D, Max Pool, Depthwise Conv, MoE, MoE Load Balance |
| **Training** | Adam, Cosine LR, Gradient Clipping, Gradient Accumulation, Mixed Precision, Activation Checkpointing |
| **Distributed** | Tensor Parallel, FSDP, Ring Attention |
| **Inference** | KV Cache, Top-k Sampling, Beam Search, Speculative Decoding, BPE, INT8 Quantization, Paged Attention |
| **Loss & Alignment** | Cross Entropy, Label Smoothing, Focal Loss, Contrastive Loss, DPO, GRPO, PPO, Reward Model |
| **Diffusion & DiT** | Noise Schedule, DDIM Step, Flow Matching, adaLN-Zero |
| **Adaptation** | LoRA, QLoRA |
| **Reasoning** | MCTS, Multi-Token Prediction |
| **SSM** | Mamba SSM |

### Learning Paths

Pick one based on what you're working toward:

| Path | Problems | Description |
|---|---|---|
| **Transformer Internals** | 12 | Activations → Normalization → Attention → GPT-2 Block |
| **Attention & Position Encoding** | 13 | Every attention variant + RoPE, ALiBi, NTK-RoPE |
| **Train a GPT from Scratch** | 15 | Embeddings → architecture → loss → optimizer → training tricks |
| **Inference & Distributed Training** | 9 | KV cache, quantization, sampling, tensor parallel, FSDP |
| **Alignment & Agent Reasoning** | 6 | Reward model → DPO → GRPO → PPO → MCTS |
| **Vision Transformer Pipeline** | 7 | Conv → patch embedding → ViT block |
| **Diffusion Models & DiT** | 5 | Noise schedule → DDIM → flow matching → adaLN-Zero |
| **LLM Frontier Architectures** | 7 | GQA, Differential Attention, MLA, MoE, Multi-Token Prediction |

```
Not sure where to start?

Fundamentals ──→ Transformer Internals ──→ Train a GPT from Scratch
                       │                          │
                       ▼                          ▼
              Attention & Position       Inference & Distributed
                       │                          │
                       ▼                          ▼
              LLM Frontier Archs         Alignment & Reasoning
                       │
               ┌───────┴───────┐
               ▼               ▼
     Vision Transformer   Diffusion & DiT
```

---

## ⚙️ Configuration

| Variable                | Default                   | Description                           |
| ----------------------- | ------------------------- | ------------------------------------- |
| `GRADING_SERVICE_URL` | `http://localhost:8000` | Grading service URL                   |
| `DB_PATH`             | `./data/pyre.db`   | SQLite database for progress tracking |

Set in `web/.env.local` to override.

---

## 📁 Project Structure

```
pyre/
├── web/                  # Next.js frontend
│   ├── src/app/          # Pages and API routes
│   ├── src/components/   # UI components
│   └── src/lib/          # Utilities, problem data
├── grading_service/      # FastAPI backend
├── torch_judge/          # Judge engine (problem definitions + test runner)
└── package.json          # Dev scripts (runs frontend + backend concurrently)
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- **Submit a new problem** — open a PR with the problem definition and test cases in `torch_judge/`
- **Report a bug** — [open an issue](https://github.com/whwangovo/pyre-code/issues) with steps to reproduce
- **Fix a bug** — fork, fix, and submit a PR
- **Improve docs** — typos, clarifications, translations

Please open an issue first for larger changes so we can discuss the approach.

---

## ⭐ Star History

![GitHub stars](https://img.shields.io/github/stars/whwangovo/pyre-code?style=social)

[![Star History Chart](https://api.star-history.com/svg?repos=whwangovo/pyre-code&type=Date)](https://star-history.com/#whwangovo/pyre-code&Date)

---

## 🙏 Acknowledgements

Problem set and judge engine based on [TorchCode](https://github.com/duoan/TorchCode) by [duoan](https://github.com/duoan), licensed under MIT.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

