"""ALiBi Attention task."""

TASK = {
    "title": "ALiBi Attention",
    "title_zh": "ALiBi 注意力",
    "difficulty": "Medium",
    "description_en": "Implement Attention with Linear Biases (ALiBi).\n\nALiBi replaces positional embeddings with a fixed linear bias added to attention scores. Each head gets a different slope `m_h`, penalizing attention to distant tokens.\n\n**Signature:** `alibi_attention(Q, K, V, num_heads) -> Tensor`\n\n**Parameters:**\n- `Q, K, V` — tensors of shape `(B, S, D)`\n- `num_heads` — number of attention heads\n\n**Returns:** output tensor `(B, S, D)`\n\n**Slope schedule:** `m_h = 1 / 2^(8h/H)` for h = 1..H\n\n**Bias:** `bias[h, i, j] = -m_h * |i - j|` (added to attention scores before softmax)",
    "description_zh": "实现带线性偏置的注意力（ALiBi）。\n\nALiBi 用固定线性偏置替代位置嵌入，直接加到注意力分数上。每个头有不同的斜率 `m_h`，对远距离 token 的注意力施加惩罚。\n\n**签名:** `alibi_attention(Q, K, V, num_heads) -> Tensor`\n\n**参数:**\n- `Q, K, V` — 形状 `(B, S, D)` 的张量\n- `num_heads` — 注意力头数\n\n**返回:** 输出张量 `(B, S, D)`\n\n**斜率:** `m_h = 1 / 2^(8h/H)`，h = 1..H\n\n**偏置:** `bias[h, i, j] = -m_h * |i - j|`（在 softmax 前加到注意力分数上）",
    "function_name": "alibi_attention",
    "hint": "1. slopes: `m_h = 2**(-8*h/H)` for h=1..H\n2. distance matrix: `|i-j|`, shape `(S,S)`\n3. bias: `-slopes[:, None, None] * distances`, shape `(H,S,S)`\n4. standard multi-head attention with bias added to scores",
    "hint_zh": "1. 斜率：`m_h = 2**(-8*h/H)`，h=1..H\n2. 距离矩阵：`|i-j|`，形状 `(S,S)`\n3. 偏置：`-slopes[:, None, None] * distances`，形状 `(H,S,S)`\n4. 标准多头注意力，将偏置加到 scores 上",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
B, S, D, H = 2, 8, 16, 4
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V, H)
assert out.shape == (B, S, D), f'Expected ({B},{S},{D}), got {out.shape}'
""",
        },
        {
            "name": "Slopes follow geometric schedule",
            "code": """
import torch
# With zero Q,K (uniform attention), the bias should break symmetry
# Verify by checking that output differs from standard attention
torch.manual_seed(42)
B, S, D, H = 1, 6, 8, 2
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_alibi = {fn}(Q, K, V, H)
# Standard attention (no bias) — compute manually
d_h = D // H
Qh = Q.view(B, S, H, d_h).transpose(1, 2)
Kh = K.view(B, S, H, d_h).transpose(1, 2)
Vh = V.view(B, S, H, d_h).transpose(1, 2)
scores = (Qh @ Kh.transpose(-2, -1)) / (d_h ** 0.5)
attn = torch.softmax(scores, dim=-1)
out_std = (attn @ Vh).transpose(1, 2).reshape(B, S, D)
assert not torch.allclose(out_alibi, out_std, atol=1e-3), 'ALiBi output should differ from standard attention'
""",
        },
        {
            "name": "Gradient flows",
            "code": """
import torch
B, S, D, H = 1, 4, 8, 2
Q = torch.randn(B, S, D, requires_grad=True)
K = torch.randn(B, S, D, requires_grad=True)
V = torch.randn(B, S, D, requires_grad=True)
out = {fn}(Q, K, V, H)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None
""",
        },
        {
            "name": "ALiBi slopes and bias numerical correctness",
            "code": """
import torch
torch.manual_seed(11)
B, S, D, H = 1, 5, 8, 4
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V, H)
# Reference: slopes m_h = 1/2^(8h/H) for h=1..H
d_h = D // H
h_idx = torch.arange(1, H + 1, dtype=torch.float32)
slopes = 1.0 / (2.0 ** (8.0 * h_idx / H))          # (H,)
pos = torch.arange(S).float()
dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()   # (S, S)
bias = -slopes.view(H, 1, 1) * dist.unsqueeze(0)    # (1, H, S, S)
Qh = Q.view(B, S, H, d_h).transpose(1, 2)
Kh = K.view(B, S, H, d_h).transpose(1, 2)
Vh = V.view(B, S, H, d_h).transpose(1, 2)
scores = (Qh @ Kh.transpose(-2, -1)) / (d_h ** 0.5) + bias
attn = torch.softmax(scores, dim=-1)
expected = (attn @ Vh).transpose(1, 2).reshape(B, S, D)
assert torch.allclose(out, expected, atol=1e-5), f'ALiBi numerical mismatch: max diff {(out - expected).abs().max()}'
""",
        },
    ],
    "solution": '''def alibi_attention(Q, K, V, num_heads):
    B, S, D = Q.shape
    d_h = D // num_heads

    # Compute slopes: m_h = 1/2^(8h/H) for h=1..H
    h_idx = torch.arange(1, num_heads + 1, dtype=torch.float32, device=Q.device)
    slopes = 1.0 / (2.0 ** (8.0 * h_idx / num_heads))  # (H,)

    # Distance matrix |i - j|, shape (S, S)
    pos = torch.arange(S, device=Q.device).float()
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (S, S)

    # ALiBi bias: (H, S, S)
    bias = -slopes.view(num_heads, 1, 1) * dist.unsqueeze(0)

    # Split into heads: (B, H, S, d_h)
    Qh = Q.view(B, S, num_heads, d_h).transpose(1, 2)
    Kh = K.view(B, S, num_heads, d_h).transpose(1, 2)
    Vh = V.view(B, S, num_heads, d_h).transpose(1, 2)

    scores = (Qh @ Kh.transpose(-2, -1)) / (d_h ** 0.5) + bias.unsqueeze(0)
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ Vh).transpose(1, 2).reshape(B, S, D)
    return out''',
    "demo": """torch.manual_seed(0)
B, S, D, H = 2, 6, 16, 4
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)

out = alibi_attention(Q, K, V, num_heads=H)
print("Output shape:", out.shape)

h_idx = torch.arange(1, H + 1, dtype=torch.float32)
slopes = 1.0 / (2.0 ** (8.0 * h_idx / H))
print("Slopes for 4 heads:", slopes)""",

}
