"""Multi-Head Attention task."""

TASK = {
    "title": "Multi-Head Attention",
    "title_zh": "多头注意力",
    "difficulty": "Hard",
    "description_en": "Implement Multi-Head Attention from scratch.\n\nMHA projects inputs into multiple heads, computes scaled dot-product attention per head, then concatenates and projects the results.\n\n**Signature:** `MultiHeadAttention(d_model, num_heads)`\n\n**Method:** `forward(Q, K, V) -> Tensor`\n- `Q` — query tensor (B, S_q, d_model)\n- `K` — key tensor (B, S_k, d_model)\n- `V` — value tensor (B, S_k, d_model)\n\n**Returns:** attention output (B, S_q, d_model)\n\n**Constraints:**\n- Use W_q, W_k, W_v, W_o as `nn.Linear(d_model, d_model)`\n- `d_k = d_model // num_heads`\n- Support cross-attention (S_q != S_k)",
    "description_zh": "从零实现多头注意力。\n\nMHA 将输入投影到多个头，每个头计算缩放点积注意力，然后拼接并投影结果。\n\n**签名:** `MultiHeadAttention(d_model, num_heads)`\n\n**方法:** `forward(Q, K, V) -> Tensor`\n- `Q` — 查询张量 (B, S_q, d_model)\n- `K` — 键张量 (B, S_k, d_model)\n- `V` — 值张量 (B, S_k, d_model)\n\n**返回:** 注意力输出 (B, S_q, d_model)\n\n**约束:**\n- 使用 W_q、W_k、W_v、W_o 作为 `nn.Linear(d_model, d_model)`\n- `d_k = d_model // num_heads`\n- 支持交叉注意力（S_q != S_k）",
    "function_name": "MultiHeadAttention",
    "hint": (
        "1. Project Q/K/V with `nn.Linear(d_model, d_model)`\n"
        "2. Reshape to `(B, heads, S, d_k)` via `.view(...).transpose(1,2)`\n"
        "3. `scores = Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`\n"
        "4. Transpose + reshape → `W_o` projection"
    ),
    "hint_zh": (
        "1. 用 `nn.Linear(d_model, d_model)` 投影 Q/K/V\n"
        "2. `.view(...).transpose(1,2)` → `(B, heads, S, d_k)`\n"
        "3. `scores = Q @ K.T / sqrt(d_k)` → `softmax` → `@ V`\n"
        "4. transpose + reshape → `W_o` 投影"
    ),
    "tests": [
        {
            "name": "Is nn.Module",
            "code": "\nimport torch, torch.nn as nn\nmha = {fn}(d_model=16, num_heads=2)\nassert isinstance(mha, nn.Module), 'MultiHeadAttention should inherit from nn.Module'\n",
        },
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
B, S, D, H = 2, 6, 32, 4
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(B, S, D)
out = mha.forward(x, x, x)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'
""",
        },
        {
            "name": "Uses nn.Linear with correct shapes",
            "code": """
import torch, torch.nn as nn
mha = {fn}(d_model=32, num_heads=4)
for name in ['W_q', 'W_k', 'W_v', 'W_o']:
    layer = getattr(mha, name)
    assert isinstance(layer, nn.Linear), f'{name} should be nn.Linear, got {type(layer)}'
    assert layer.weight.shape == (32, 32), f'{name}.weight shape: {layer.weight.shape}'
    assert layer.weight.requires_grad, f'{name}.weight must require grad'
""",
        },
        {
            "name": "Numerical correctness vs reference",
            "code": """
import torch, torch.nn as nn, math
torch.manual_seed(0)
D, H = 16, 2
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
Q = torch.randn(1, 4, D)
K = torch.randn(1, 4, D)
V = torch.randn(1, 4, D)
out = mha.forward(Q, K, V)
q = mha.W_q(Q).view(1, 4, H, d_k).transpose(1, 2)
k = mha.W_k(K).view(1, 4, H, d_k).transpose(1, 2)
v = mha.W_v(V).view(1, 4, H, d_k).transpose(1, 2)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
ref = mha.W_o(attn.transpose(1, 2).contiguous().view(1, 4, D))
assert torch.allclose(out, ref, atol=1e-5), 'Output does not match reference'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(0)
mha = {fn}(d_model=16, num_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
out = mha.forward(x, x, x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert mha.W_q.weight.grad is not None, 'W_q.weight.grad is None'
assert mha.W_o.weight.grad is not None, 'W_o.weight.grad is None'
""",
        },
        {
            "name": "Cross-attention (seq_q != seq_k)",
            "code": """
import torch
mha = {fn}(d_model=32, num_heads=4)
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out = mha.forward(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'
""",
        },
        {
            "name": "Different heads give different outputs",
            "code": """
import torch
torch.manual_seed(42)
D, H = 16, 4
d_k = D // H
mha = {fn}(d_model=D, num_heads=H)
x = torch.randn(1, 4, D)
q = mha.W_q(x).view(1, 4, H, d_k).transpose(1, 2)
assert not torch.allclose(q[:, 0], q[:, 1], atol=1e-3), 'Heads produce identical queries'
""",
        },
    ],
    "solution": '''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]

        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)

        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.W_o(out)''',
    "demo": """torch.manual_seed(0)
mha = MultiHeadAttention(d_model=32, num_heads=4)
x = torch.randn(2, 6, 32)
out = mha.forward(x, x, x)
print("Self-attn shape:", out.shape)

Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out2 = mha.forward(Q, K, V)
print("Cross-attn shape:", out2.shape)""",

}