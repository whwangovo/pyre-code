"""Mamba Selective State Space Model step task."""

TASK = {
    "title": "Mamba SSM Step",
    "title_zh": "Mamba SSM 步骤",
    "difficulty": "Hard",
    "description_en": "Implement a single recurrent step of the Mamba selective state space model (SSM).\n\nMamba's key innovation is that B, C, and delta are **input-dependent** (selective), unlike classical SSMs where they are fixed. This allows the model to selectively remember or forget information.\n\n**Signature:** `mamba_ssm_step(x, h, A, B, C, delta) -> (y, h_new)`\n\n**Parameters:**\n- `x` — input at current step, shape (B, D)\n- `h` — hidden state, shape (B, D, N)\n- `A` — state transition, shape (D, N) — used as `A = -exp(A_log)` upstream\n- `B` — input projection (selective), shape (B, D, N)\n- `C` — output projection (selective), shape (B, D, N)\n- `delta` — discretization step (selective, after softplus), shape (B, D)\n\n**Discretization (Zero-Order Hold):**\n- `dA = exp(delta.unsqueeze(-1) * A)` — shape (B, D, N)\n- `dB = delta.unsqueeze(-1) * B` — shape (B, D, N)\n\n**Recurrence:**\n- `h_new = dA * h + dB * x.unsqueeze(-1)`\n- `y = (h_new * C).sum(dim=-1)` — shape (B, D)\n\n**Returns:** tuple `(y, h_new)`",
    "description_zh": "实现 Mamba 选择性状态空间模型（SSM）的单步递推。\n\nMamba 的核心创新在于 B、C 和 delta 是**输入相关的**（选择性的），不同于经典 SSM 中的固定参数。这使模型能够选择性地记忆或遗忘信息。\n\n**签名:** `mamba_ssm_step(x, h, A, B, C, delta) -> (y, h_new)`\n\n**参数:**\n- `x` — 当前步输入，形状 (B, D)\n- `h` — 隐藏状态，形状 (B, D, N)\n- `A` — 状态转移矩阵，形状 (D, N)（上游使用 `A = -exp(A_log)`）\n- `B` — 输入投影（选择性），形状 (B, D, N)\n- `C` — 输出投影（选择性），形状 (B, D, N)\n- `delta` — 离散化步长（选择性，经过 softplus），形状 (B, D)\n\n**离散化（零阶保持）:**\n- `dA = exp(delta.unsqueeze(-1) * A)` — 形状 (B, D, N)\n- `dB = delta.unsqueeze(-1) * B` — 形状 (B, D, N)\n\n**递推:**\n- `h_new = dA * h + dB * x.unsqueeze(-1)`\n- `y = (h_new * C).sum(dim=-1)` — 形状 (B, D)\n\n**返回:** 元组 `(y, h_new)`",
    "function_name": "mamba_ssm_step",
    "hint": "1. `dA = exp(delta.unsqueeze(-1) * A)`  shape `(B,D,N)`\n2. `dB = delta.unsqueeze(-1) * B`\n3. `h_new = dA * h + dB * x.unsqueeze(-1)`\n4. `y = (h_new * C).sum(dim=-1)`  shape `(B,D)`",
    "hint_zh": "1. `dA = exp(delta.unsqueeze(-1) * A)`  形状 `(B,D,N)`\n2. `dB = delta.unsqueeze(-1) * B`\n3. `h_new = dA * h + dB * x.unsqueeze(-1)`\n4. `y = (h_new * C).sum(dim=-1)`  形状 `(B,D)`",
    "tests": [
        {
            "name": "Output shapes",
            "code": "\nimport torch\nB_sz, D, N = 2, 8, 4\nx = torch.randn(B_sz, D)\nh = torch.randn(B_sz, D, N)\nA = torch.randn(D, N)\nB = torch.randn(B_sz, D, N)\nC = torch.randn(B_sz, D, N)\ndelta = torch.rand(B_sz, D) + 0.1\ny, h_new = {fn}(x, h, A, B, C, delta)\nassert y.shape == (B_sz, D), f'y shape: {y.shape}'\nassert h_new.shape == (B_sz, D, N), f'h_new shape: {h_new.shape}'\n"
        },
        {
            "name": "h_new depends on both h and x",
            "code": "\nimport torch\ntorch.manual_seed(0)\nB_sz, D, N = 2, 4, 3\nx = torch.randn(B_sz, D)\nh = torch.randn(B_sz, D, N)\nA = -torch.rand(D, N)  # negative for stability\nB = torch.randn(B_sz, D, N)\nC = torch.randn(B_sz, D, N)\ndelta = torch.rand(B_sz, D) + 0.1\n_, h1 = {fn}(x, h, A, B, C, delta)\n_, h2 = {fn}(x, torch.zeros_like(h), A, B, C, delta)\n_, h3 = {fn}(torch.zeros_like(x), h, A, B, C, delta)\nassert not torch.allclose(h1, h2), 'h_new must depend on h'\nassert not torch.allclose(h1, h3), 'h_new must depend on x'\n"
        },
        {
            "name": "Gradient flows through x, B, C, delta",
            "code": """
import torch
B_sz, D, N = 2, 4, 3
x = torch.randn(B_sz, D, requires_grad=True)
h = torch.randn(B_sz, D, N)
A = torch.randn(D, N)
Bt = torch.randn(B_sz, D, N, requires_grad=True)
C = torch.randn(B_sz, D, N, requires_grad=True)
delta = torch.ones(B_sz, D, requires_grad=True)  # leaf tensor, no +0.1 op
y, h_new = {fn}(x, h, A, Bt, C, delta)
(y.sum() + h_new.sum()).backward()
assert x.grad is not None, 'x.grad is None'
assert Bt.grad is not None, 'B.grad is None'
assert C.grad is not None, 'C.grad is None'
assert delta.grad is not None, 'delta.grad is None'
""",
        },
        {
            "name": "With A=-inf, h_new equals dB * x (no memory)",
            "code": "\nimport torch\ntorch.manual_seed(1)\nB_sz, D, N = 1, 4, 3\nx = torch.randn(B_sz, D)\nh = torch.randn(B_sz, D, N)\nA = torch.full((D, N), -1e9)  # exp(-inf * delta) ~ 0\nB = torch.randn(B_sz, D, N)\nC = torch.randn(B_sz, D, N)\ndelta = torch.rand(B_sz, D) + 0.1\n_, h_new = {fn}(x, h, A, B, C, delta)\nexpected = delta.unsqueeze(-1) * B * x.unsqueeze(-1)\nassert torch.allclose(h_new, expected, atol=1e-4), 'With A=-inf, h_new should be dB*x only'\n"
        },
        {
            "name": "Numerical correctness of dA formula",
            "code": "\nimport torch\ntorch.manual_seed(3)\nB_sz, D, N = 2, 4, 3\nx = torch.randn(B_sz, D)\nh = torch.zeros(B_sz, D, N)\nA = -torch.rand(D, N)\nB = torch.randn(B_sz, D, N)\nC = torch.randn(B_sz, D, N)\ndelta = torch.rand(B_sz, D) + 0.1\n_, h_new = {fn}(x, h, A, B, C, delta)\ndA_ref = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))\ndB_ref = delta.unsqueeze(-1) * B\nexpected_h = dA_ref * h + dB_ref * x.unsqueeze(-1)\nassert torch.allclose(h_new, expected_h, atol=1e-5), 'h_new does not match ZOH formula'\n"
        }
    ],
    "solution": '''def mamba_ssm_step(x, h, A, B, C, delta):
    # delta: (B, D), A: (D, N), B: (B, D, N), C: (B, D, N)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))   # (B, D, N)
    dB = delta.unsqueeze(-1) * B                            # (B, D, N)
    h_new = dA * h + dB * x.unsqueeze(-1)                  # (B, D, N)
    y = (h_new * C).sum(dim=-1)                             # (B, D)
    return y, h_new''',
    "demo": """torch.manual_seed(0)
B_size, D, N = 2, 8, 4  # batch, channels, state_dim

x     = torch.randn(B_size, D)
h     = torch.zeros(B_size, D, N)
A     = -torch.rand(D, N)          # negative for stability
B_mat = torch.randn(D, N)
C     = torch.randn(B_size, D, N)
delta = torch.rand(B_size, D).add(0.1)  # positive step sizes

y, h_new = mamba_ssm_step(x, h, A, B_mat, C, delta)

print("y shape:    ", y.shape)      # (2, 8)
print("h_new shape:", h_new.shape)  # (2, 8, 4)

dA_manual = torch.exp(delta[0, 0] * A[0])
dA_check  = torch.exp(delta[0:1, 0:1] * A[0:1])[0, 0]
print("dA formula check (should be ~0):", (dA_manual - dA_check).abs().max().item())""",

}
