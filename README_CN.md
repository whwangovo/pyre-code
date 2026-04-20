[English](./README.md) | [中文](./README_CN.md)

<p align="center">
  <h1 align="center">🔥 Pyre Code</h1>
  <p align="center">
    从零实现 Transformers、vLLM、TRL 等系统的核心模块。
  </p>
  <p align="center">
    <em>读完论文，写出代码。无需 GPU。</em>
  </p>
  <p align="center">
    <a href="https://star-history.com/#whwangovo/pyre-code&Date">
      <img src="https://img.shields.io/github/stars/whwangovo/pyre-code?style=social" alt="GitHub stars" />
    </a>
  </p>
</p>

---

## 🧠 这是什么？

76 道实现题，覆盖注意力机制、训练技巧、推理优化、对齐算法、图神经网络等核心模块。你写代码，本地评测跑测试，红了就改，绿了就过。不需要 GPU。

### 谁适合用？

- **备战 ML 面试** — 手写核心组件，不只是背概念
- **偏好动手学习** — 写代码比看视频更容易理解
- **深入理解原理** — 用过 `nn.MultiheadAttention`，现在自己实现一个

### 功能亮点

- **浏览器直接写** — 内置 Monaco 编辑器，开箱即用，不用折腾本地 IDE
- **秒级反馈** — 提交即判，逐条显示测试结果
- **参考实现** — 先自己写，再看答案
- **进度记录** — 做了多少、试了几次，关掉浏览器也不丢
- **AI 辅助** — 可选的 AI 提示功能，支持任意 OpenAI 兼容 API（在 `.env` 或 UI 中配置）
- **数据不出本机** — 全部本地运行，没有任何远程调用（除非你启用了 AI 辅助）

### 技术栈

| 层级     | 技术                                                                   |
| -------- | ---------------------------------------------------------------------- |
| 前端     | Next.js + Monaco Editor + Tailwind CSS                                 |
| 后端     | FastAPI 评测服务                                                       |
| 评测引擎 | [torch_judge](https://github.com/duoan/TorchCode) — 执行并验证提交的代码 |
| 存储     | SQLite（进度持久化）                                                   |

---

## 📢 最新动态

- **[2026/04/20]** 新增 GNN 学习路径 — 8 道题覆盖 GCN、GAT、GIN、MPNN、GraphSAGE、链接预测、图自编码器。🔥
- **[2026/04/20]** 全新 UI 设计，采用 OKLch 色彩系统、暗色模式和 Geist 字体 — 可一键切换回经典设计。🔥
- **[2026/04/13]** 提交历史 — 查看每道题的所有历史提交记录。
- **[2026/04/10]** AI 辅助 — 可选的 AI 提示功能，支持任意 OpenAI 兼容 API。🔥
- **[2026/04/10]** print 输出捕获 — `print()` 的内容会显示在测试结果中。
- **[2026/04/09]** 68 道题，覆盖 Transformers、vLLM、TRL、扩散模型等核心模块。🔥
- **[2026/04/09]** Pyre Code 正式发布 🎉

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+

### 安装

**方式 A — 一键启动（推荐）**

macOS / Linux:

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
./setup.sh
npm run dev
```

Windows (PowerShell):

```powershell
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
.\setup.ps1
npm run dev
```

Windows (CMD):

```cmd
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
setup.bat
npm run dev
```

安装脚本会自动创建 `.venv` 虚拟环境（优先用 `uv`，没有就回退到 `python -m venv`），装好所有依赖。

如果仓库里存在 `.venv`，`npm run dev` 会优先使用项目内 Python；如果没有，再回退到当前 shell 里的 `python`。

**方式 B — conda**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
conda create -n pyre python=3.11 -y && conda activate pyre
pip install -e ".[dev]"
npm install
npm run dev   # 记得先激活 conda 环境
```

**方式 C — 手动（venv）**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code

# 二选一：
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
# 或者：python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
# Windows：python -m venv .venv && .venv\Scripts\activate && pip install -e ".[dev]"

npm install
npm run dev
```

跑起来之后：

- **评测服务** → `http://localhost:8000`
- **Web 界面** → `http://localhost:3000`

**方式 D — Docker**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre-code
docker compose up --build
```

打开 `http://localhost:3000` 即可。进度存在 Docker 卷里，`docker compose down -v` 可以重置。

### AI 辅助（可选）

如需启用服务端 AI 提示，将 `web/.env.example` 复制为 `web/.env` 并填入：

```bash
AI_HELP_BASE_URL=https://api.openai.com/v1
AI_HELP_API_KEY=sk-...
AI_HELP_MODEL=gpt-4o-mini
```

支持任意 OpenAI 兼容接口（OpenAI、Anthropic 代理、Ollama 等）。如果未配置服务端，用户也可以在 UI 中自行填写 API key。

---

## 📋 题目一览

76 道题，按方向分组：

| 方向                 | 题目                                                                                               |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| **基础**       | ReLU、Softmax、GELU、SwiGLU、Dropout、Embedding、Linear、Kaiming 初始化、线性回归                  |
| **归一化**     | LayerNorm、BatchNorm、RMSNorm                                                                      |
| **注意力**     | 缩放点积、多头、因果、交叉、GQA、滑动窗口、线性、Flash、差分注意力、MLA                            |
| **位置编码**   | 正弦编码、RoPE、ALiBi、NTK-aware RoPE                                                              |
| **架构**       | SwiGLU MLP、GPT-2 Block、ViT Patch、ViT Block、Conv2D、Max Pool、深度可分离卷积、MoE、MoE 负载均衡 |
| **训练**       | Adam、余弦学习率、梯度裁剪、梯度累积、混合精度、激活检查点                                         |
| **分布式**     | 张量并行、FSDP、环形注意力                                                                         |
| **推理**       | KV Cache、Top-k 采样、束搜索、推测解码、BPE、INT8 量化、分页注意力                                 |
| **损失与对齐** | 交叉熵、标签平滑、Focal Loss、对比损失、DPO、GRPO、PPO、奖励模型                                   |
| **扩散与 DiT** | 噪声调度、DDIM 步骤、流匹配、adaLN-Zero                                                            |
| **适配**       | LoRA、QLoRA                                                                                        |
| **推理搜索**   | MCTS、多 Token 预测                                                                                |
| **SSM**        | Mamba SSM                                                                                          |
| **图神经网络** | GCN、Graph Readout、GAT、GIN、MPNN、GraphSAGE、链接预测、图自编码器                                |

### 学习路径

不知道从哪下手？挑一条适合自己的：

| 路径                           | 题数 | 覆盖内容                                        |
| ------------------------------ | ---- | ----------------------------------------------- |
| **Transformer 内部机制** | 12   | 激活函数 → 归一化 → 注意力 → GPT-2 Block     |
| **注意力与位置编码**     | 13   | 所有注意力变体 + RoPE、ALiBi、NTK-RoPE          |
| **从零训练 GPT**         | 15   | Embedding → 架构 → 损失 → 优化器 → 训练技巧 |
| **推理与分布式**         | 9    | KV Cache、量化、采样、张量并行、FSDP            |
| **对齐与推理搜索**       | 6    | 奖励模型 → DPO → GRPO → PPO → MCTS          |
| **ViT 全流程**           | 7    | 卷积 → Patch Embedding → ViT Block            |
| **扩散模型与 DiT**       | 5    | 噪声调度 → DDIM → 流匹配 → adaLN-Zero        |
| **LLM 前沿架构**         | 7    | GQA、差分注意力、MLA、MoE、多 Token 预测        |
| **图神经网络**           | 8    | GCN → GAT → GIN → MPNN → GraphSAGE → 链接预测 → GAE |

```
路径导航：

基础 ──→ Transformer 内部机制 ──→ 从零训练 GPT
                 │                        │
                 ▼                        ▼
        注意力与位置编码            推理与分布式
                 │                        │
                 ▼                        ▼
        LLM 前沿架构             对齐与推理搜索
                 │
          ┌──────┼──────┐
          ▼      ▼      ▼
     ViT 全流程  扩散   图神经网络
```

---

## ⚙️ 配置

| 环境变量                | 默认值                    | 说明              |
| ----------------------- | ------------------------- | ----------------- |
| `GRADING_SERVICE_URL` | `http://localhost:8000` | 评测服务地址      |
| `DB_PATH`             | `./data/pyre.db`        | SQLite 数据库路径 |

在 `web/.env.local` 中设置即可覆盖。

---

## 📁 项目结构

```
pyre-code/
├── web/                  # Next.js 前端
│   ├── src/app/          # 页面与 API 路由
│   ├── src/components/   # UI 组件
│   └── src/lib/          # 工具函数、题目数据
├── grading_service/      # FastAPI 评测后端
├── torch_judge/          # 评测引擎（题目定义 + 测试执行）
└── package.json          # 开发脚本（前后端并行启动）
```

---

## 🤝 参与贡献

- **出新题** — 在 `torch_judge/` 里添加题目定义和测试用例，提 PR
- **报 Bug** — [开 issue](https://github.com/whwangovo/pyre-code/issues)，附上复现步骤
- **修 Bug** — fork → 修复 → PR
- **改文档** — 错别字、表述不清、翻译，都欢迎

大改动建议先开 issue 聊聊思路。

---

## ⭐ Star History

![GitHub stars](https://img.shields.io/github/stars/whwangovo/pyre-code?style=social)

[![Star History Chart](https://api.star-history.com/svg?repos=whwangovo/pyre-code&type=Date)](https://star-history.com/#whwangovo/pyre-code&Date)

---

## 🙏 致谢

题库和评测引擎基于 [duoan](https://github.com/duoan) 的 [TorchCode](https://github.com/duoan/TorchCode)，MIT 协议。

---

## 📄 许可证

MIT License，详见 [LICENSE](LICENSE)。
