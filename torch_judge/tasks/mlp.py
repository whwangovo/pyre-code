"""SwiGLU MLP task."""

TASK = {
    "title": "SwiGLU MLP",
    "title_zh": "SwiGLU MLP",
    "difficulty": "Medium",
    "description_en": "Implement a SwiGLU MLP as an nn.Module.\n\nSwiGLU is a gated MLP variant used in LLaMA and other modern LLMs: it applies SiLU gating to one projection and element-wise multiplies with another.\n\n**Signature:** `SwiGLUMLP(d_model, d_ff)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (*, d_model)\n\n**Returns:** output tensor (*, d_model)\n\n**Constraints:**\n- Three projections: gate_proj(d, d_ff), up_proj(d, d_ff), down_proj(d_ff, d)\n- `forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))`\n- SiLU(x) = x * sigmoid(x)",
    "description_zh": "实现 SwiGLU MLP（nn.Module）。\n\nSwiGLU 是 LLaMA 等现代 LLM 使用的门控 MLP 变体：对一个投影应用 SiLU 门控，与另一个投影逐元素相乘。\n\n**签名:** `SwiGLUMLP(d_model, d_ff)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (*, d_model)\n\n**返回:** 输出张量 (*, d_model)\n\n**约束:**\n- 三个投影：gate_proj(d, d_ff)、up_proj(d, d_ff)、down_proj(d_ff, d)\n- `forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))`\n- SiLU(x) = x * sigmoid(x)",
    "function_name": "SwiGLUMLP",
    "hint": "`gate_proj(d→d_ff)`, `up_proj(d→d_ff)`, `down_proj(d_ff→d)`. Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`. `silu(x) = x * sigmoid(x)`.",
    "hint_zh": "`gate_proj(d→d_ff)`、`up_proj(d→d_ff)`、`down_proj(d_ff→d)`。前向：`down_proj(silu(gate_proj(x)) * up_proj(x))`。`silu(x) = x * sigmoid(x)`。",
    "tests": [
        {
            "name": "Parameter shapes",
            "code": """
import torch, torch.nn as nn
mlp = {fn}(d_model=64, d_ff=128)
assert isinstance(mlp, nn.Module), 'Must inherit from nn.Module'
assert hasattr(mlp, 'gate_proj'), 'Need self.gate_proj'
assert hasattr(mlp, 'up_proj'), 'Need self.up_proj'
assert hasattr(mlp, 'down_proj'), 'Need self.down_proj'
assert mlp.gate_proj.weight.shape == (128, 64), f'gate_proj shape: {mlp.gate_proj.weight.shape}'
assert mlp.up_proj.weight.shape == (128, 64), f'up_proj shape: {mlp.up_proj.weight.shape}'
assert mlp.down_proj.weight.shape == (64, 128), f'down_proj shape: {mlp.down_proj.weight.shape}'
""",
        },
        {
            "name": "Forward output shape",
            "code": """
import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(2, 8, 32)
out = mlp(x)
assert out.shape == (2, 8, 32), f'Output shape: {out.shape}'
""",
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch, torch.nn.functional as F
torch.manual_seed(0)
mlp = {fn}(d_model=16, d_ff=32)
x = torch.randn(1, 4, 16)
out = mlp(x)
gate = mlp.gate_proj(x)
up = mlp.up_proj(x)
ref = mlp.down_proj(F.silu(gate) * up)
assert torch.allclose(out, ref, atol=1e-5), 'Output != down(silu(gate(x)) * up(x))'
""",
        },
        {
            "name": "2-D input",
            "code": """
import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(4, 32)
out = mlp(x)
assert out.shape == (4, 32), f'2-D output shape: {out.shape}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(2, 4, 32, requires_grad=True)
mlp(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in mlp.parameters())
n_grad = sum(1 for p in mlp.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
        },
    ],
    "solution": '''class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gate = self.gate_proj(x)
        silu_gate = gate * torch.sigmoid(gate)
        return self.down_proj(silu_gate * self.up_proj(x))''',
    "demo": """mlp = SwiGLUMLP(d_model=64, d_ff=128)
x = torch.randn(2, 8, 64)
print('Output:', mlp(x).shape)
print('Params:', sum(p.numel() for p in mlp.parameters()))""",

}