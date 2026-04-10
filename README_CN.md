[English](./README.md) | [中文](./README_CN.md)

<p align="center">
  <h1 align="center">🔥 Pyre Code</h1>
  <p align="center">
    从零复现现代 AI 系统的内核——Transformers、vLLM、TRL，以及更多。
    <br />
    顶级 ML 工程团队面试中考察的核心技能。
  </p>
</p>

---

## 这是什么？

68 道题。你写实现，本地评测服务跑测试，看哪里挂了。就这样。

题目覆盖 Transformers、vLLM、TRL 和扩散模型的实际内核——注意力变体、训练技巧、推理算法、对齐算法。无需 GPU。

### 特性

- **浏览器编辑器** — Monaco + Python 语法高亮，无需配置 IDE
- **即时反馈** — 提交后秒级返回每个测试用例的通过/失败
- **参考答案** — 自己做完再对比
- **进度追踪** — 解题数和尝试记录，跨会话持久化
- **完全本地** — 代码不离开你的机器

### 技术栈

| 层级 | 技术 |
|---|---|
| 前端 | Next.js + Monaco Editor + Tailwind CSS |
| 后端 | FastAPI 评测服务 |
| 评测引擎 | [torch_judge](https://github.com/duoan/TorchCode) — 执行并验证提交的代码 |
| 存储 | SQLite（进度追踪） |

---

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+

### 安装

**方式 A — 一键安装（推荐）**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre
./setup.sh
npm run dev
```

`setup.sh` 自动创建并激活 Python 环境（优先级：`uv` → `conda` → `venv`），安装所有依赖，完成后打印启动命令。

**方式 B — 手动安装**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre

# 创建 Python 环境（二选一）：
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e .
# 或：python3 -m venv .venv && source .venv/bin/activate && pip install -e .

npm install
npm run dev
```

> **conda 用户：** `npm run dev` 默认使用 `.venv/bin/python`，如果你用 conda 则需要先激活环境再运行 `npm run dev`：
>
> ```bash
> conda create -n pyre python=3.11 -y && conda activate pyre
> pip install -e .
> npm install
> npm run dev   # 需在 conda 环境激活状态下运行
> ```

启动后：

- **评测服务** → `http://localhost:8000`
- **Web 界面** → `http://localhost:3000`

**方式 C — Docker**

```bash
git clone https://github.com/whwangovo/pyre-code.git
cd pyre
docker compose up --build
```

打开 `http://localhost:3000`。进度保存在 Docker 卷中。运行 `docker compose down -v` 可重置数据。

---

## 题目列表

68 道题目，按类别分组：

| 类别 | 题目 |
|---|---|
| **基础** | ReLU、Softmax、GELU、SwiGLU、Dropout、Embedding、Linear、Kaiming 初始化、线性回归 |
| **归一化** | LayerNorm、BatchNorm、RMSNorm |
| **注意力** | 缩放点积、多头、因果、交叉、GQA、滑动窗口、线性、Flash、差分注意力、MLA |
| **位置编码** | 正弦编码、RoPE、ALiBi、NTK-aware RoPE |
| **架构** | SwiGLU MLP、GPT-2 Block、ViT Patch、ViT Block、Conv2D、Max Pool、深度可分离卷积、MoE、MoE 负载均衡 |
| **训练** | Adam、余弦学习率、梯度裁剪、梯度累积、混合精度、激活检查点 |
| **分布式** | 张量并行、FSDP、环形注意力 |
| **推理** | KV Cache、Top-k 采样、束搜索、推测解码、BPE、INT8 量化、分页注意力 |
| **损失与对齐** | 交叉熵、标签平滑、Focal Loss、对比损失、DPO、GRPO、PPO、奖励模型 |
| **扩散与 DiT** | 噪声调度、DDIM 步骤、流匹配、adaLN-Zero |
| **适配** | LoRA、QLoRA |
| **推理搜索** | MCTS、多 Token 预测 |
| **SSM** | Mamba SSM |

### 学习路径

按目标选一条：

| 路径 | 题目数 | 内容 |
|---|---|---|
| **Transformer 内部机制** | 12 | 激活函数 → 归一化 → 注意力 → GPT-2 Block |
| **注意力与位置编码** | 13 | 全部注意力变体 + RoPE、ALiBi、NTK-RoPE |
| **从零训练 GPT** | 15 | Embedding → 架构 → 损失 → 优化器 → 训练技巧 |
| **推理与分布式训练** | 9 | KV Cache、量化、采样、张量并行、FSDP |
| **对齐与智能体推理** | 6 | 奖励模型 → DPO → GRPO → PPO → MCTS |
| **Vision Transformer 流水线** | 7 | 卷积 → Patch Embedding → ViT Block |
| **扩散模型与 DiT** | 5 | 噪声调度 → DDIM → 流匹配 → adaLN-Zero |
| **LLM 前沿架构** | 7 | GQA、差分注意力、MLA、MoE、多 Token 预测 |

---

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GRADING_SERVICE_URL` | `http://localhost:8000` | 评测服务地址 |
| `DB_PATH` | `./data/pyre.db` | 进度追踪用 SQLite 数据库路径 |

在 `web/.env.local` 中设置以覆盖默认值。

---

## 项目结构

```
pyre/
├── web/                  # Next.js 前端
│   ├── src/app/          # 页面和 API 路由
│   ├── src/components/   # UI 组件
│   └── src/lib/          # 工具函数、题目数据
├── grading_service/      # FastAPI 后端
├── torch_judge/          # 评测引擎（题目定义 + 测试运行器）
└── package.json          # 开发脚本（同时启动前后端）
```

---

## 致谢

题库和评测引擎基于 [duoan](https://github.com/duoan) 的 [TorchCode](https://github.com/duoan/TorchCode)，遵循 MIT 协议。

本项目在原版基础上新增了 Web 前端和 REST 评测服务，作为原 Jupyter 界面的替代方案。

---

## 许可证

本项目基于 MIT 许可证分发。详见 [LICENSE](LICENSE)。

