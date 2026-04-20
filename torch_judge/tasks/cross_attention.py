"""Multi-Head Cross-Attention task."""

TASK = {
    "title": "Multi-Head Cross-Attention",
    "title_zh": "多头交叉注意力",
    "difficulty": "Medium",
    "description_en": "Implement multi-head cross-attention as an nn.Module.\n\nCross-attention lets a decoder attend to encoder outputs: Q comes from one sequence, K/V from another. No causal mask is applied.\n\n**Signature:** `MultiHeadCrossAttention(d_model, num_heads)` (nn.Module)\n\n**Forward:** `forward(x_q, x_kv) -> Tensor`\n- `x_q` — query input (B, S_q, d_model)\n- `x_kv` — key/value input (B, S_kv, d_model)\n\n**Returns:** attention output (B, S_q, d_model)\n\n**Constraints:**\n- Use separate W_q, W_k, W_v, W_o linear projections\n- Q and KV can have different sequence lengths",
    "description_zh": "实现多头交叉注意力（nn.Module）。\n\n交叉注意力让解码器关注编码器输出：Q 来自一个序列，K/V 来自另一个序列，不使用因果掩码。\n\n**签名:** `MultiHeadCrossAttention(d_model, num_heads)`（nn.Module）\n\n**前向传播:** `forward(x_q, x_kv) -> Tensor`\n- `x_q` — 查询输入 (B, S_q, d_model)\n- `x_kv` — 键/值输入 (B, S_kv, d_model)\n\n**返回:** 注意力输出 (B, S_q, d_model)\n\n**约束:**\n- 使用独立的 W_q、W_k、W_v、W_o 线性投影\n- Q 和 KV 可以有不同的序列长度",
    "function_name": "MultiHeadCrossAttention",
    "hint": "Q from `x_q`, K/V from `x_kv`. Project → reshape to `(B, H, S, d_k)` → scaled dot-product (no causal mask) → concat heads → `W_o`.",
    "hint_zh": "Q 来自 `x_q`，K/V 来自 `x_kv`。投影 → reshape 为 `(B, H, S, d_k)` → 缩放点积（无因果遮蔽）→ 拼接各头 → `W_o`。",
    "tests": [
        {
            "name": "Output shape",
            "code": "\nimport torch, torch.nn as nn\nattn = {fn}(d_model=64, num_heads=4)\nassert isinstance(attn, nn.Module), 'Must inherit from nn.Module'\nout = attn(torch.randn(2, 6, 64), torch.randn(2, 10, 64))\nassert out.shape == (2, 6, 64), f'Output shape: {out.shape}'\n"
        },
        {
            "name": "Q and KV different lengths",
            "code": "\nimport torch\nattn = {fn}(d_model=32, num_heads=2)\nout = attn(torch.randn(1, 3, 32), torch.randn(1, 20, 32))\nassert out.shape == (1, 3, 32), f'Shape: {out.shape}'\n"
        },
        {
            "name": "No causal mask \u2014 all KV affects all Q",
            "code": "\nimport torch\ntorch.manual_seed(0)\nattn = {fn}(d_model=32, num_heads=2)\nx_q = torch.randn(1, 4, 32)\nx_kv = torch.randn(1, 6, 32)\nout1 = attn(x_q, x_kv)\nx_kv2 = x_kv.clone()\nx_kv2[:, -1] = torch.randn(1, 32)\nout2 = attn(x_q, x_kv2)\nassert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Changing last KV should affect all Q positions'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nattn = {fn}(d_model=32, num_heads=2)\nx_q = torch.randn(1, 4, 32, requires_grad=True)\nx_kv = torch.randn(1, 6, 32, requires_grad=True)\nattn(x_q, x_kv).sum().backward()\nassert x_q.grad is not None and x_kv.grad is not None, 'Missing gradients'\n"
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(5)
B, S_q, S_kv, D, num_heads = 2, 4, 6, 32, 4
attn = {fn}(d_model=D, num_heads=num_heads)
attn.eval()
x_q  = torch.randn(B, S_q,  D)
x_kv = torch.randn(B, S_kv, D)
with torch.no_grad():
    out = attn(x_q, x_kv)
    d_k = D // num_heads
    # Project using module weights
    Q = x_q  @ attn.W_q.weight.T + attn.W_q.bias   # (B, S_q,  D)
    K = x_kv @ attn.W_k.weight.T + attn.W_k.bias   # (B, S_kv, D)
    V = x_kv @ attn.W_v.weight.T + attn.W_v.bias   # (B, S_kv, D)
    # Split heads: (B, H, S, d_k)
    Q = Q.view(B, S_q,  num_heads, d_k).transpose(1, 2)
    K = K.view(B, S_kv, num_heads, d_k).transpose(1, 2)
    V = V.view(B, S_kv, num_heads, d_k).transpose(1, 2)
    scores  = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    ctx     = (weights @ V).transpose(1, 2).contiguous().view(B, S_q, D)
    expected = ctx @ attn.W_o.weight.T + attn.W_o.bias
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff {(out - expected).abs().max()}'
""",
        },
    ],
    "solution": '''class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        B, S_q, _ = x_q.shape
        S_kv = x_kv.shape[1]
        q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S_q, -1))''',
    "demo": """attn = MultiHeadCrossAttention(64, 4)
x_q = torch.randn(2, 6, 64)
x_kv = torch.randn(2, 10, 64)
print('Output:', attn(x_q, x_kv).shape)""",

}