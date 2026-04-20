"""Causal Self-Attention task."""

TASK = {
    "title": "Causal Self-Attention",
    "title_zh": "因果自注意力",
    "difficulty": "Medium",
    "description_en": "Implement causal (masked) self-attention.\n\nLike standard attention but prevents each position from attending to future positions, essential for autoregressive models like GPT.\n\n**Signature:** `causal_attention(Q, K, V) -> Tensor`\n\n**Parameters:**\n- `Q` — query tensor (B, S, D)\n- `K` — key tensor (B, S, D)\n- `V` — value tensor (B, S, D)\n\n**Returns:** causally masked attention output (B, S, D)\n\n**Constraints:**\n- Mask future positions with `-inf` before softmax\n- Position 0 should only see itself",
    "description_zh": "实现因果（掩码）自注意力。\n\n与标准注意力类似，但阻止每个位置关注未来位置，这对 GPT 等自回归模型至关重要。\n\n**签名:** `causal_attention(Q, K, V) -> Tensor`\n\n**参数:**\n- `Q` — 查询张量 (B, S, D)\n- `K` — 键张量 (B, S, D)\n- `V` — 值张量 (B, S, D)\n\n**返回:** 因果掩码注意力输出 (B, S, D)\n\n**约束:**\n- 在 softmax 之前用 `-inf` 掩盖未来位置\n- 位置 0 只能看到自身",
    "function_name": "causal_attention",
    "hint": "Standard attention + causal mask. `torch.triu(ones, diagonal=1)` → upper triangle → fill with `-inf` before softmax.",
    "hint_zh": "标准注意力 + 因果遮蔽。`torch.triu(ones, diagonal=1)` 得到上三角 → softmax 前填 `-inf`。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
out = {fn}(torch.randn(2, 6, 16), torch.randn(2, 6, 16), torch.randn(2, 6, 16))
assert out.shape == (2, 6, 16), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Future tokens don't affect past",
            "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V)
K2, V2 = K.clone(), V.clone()
K2[:, 4:] = torch.randn(B, 4, D)
V2[:, 4:] = torch.randn(B, 4, D)
out2 = {fn}(Q, K2, V2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Changing future K/V affected past outputs'
""",
        },
        {
            "name": "First position only sees itself",
            "code": """
import torch
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V)
assert torch.allclose(out[:, 0], V[:, 0], atol=1e-5), 'Position 0 should output V[0]'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch
torch.manual_seed(42)
B, S, D = 2, 6, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
# Reference computation
d_k = Q.shape[-1]
scale = d_k ** -0.5
scores = Q @ K.transpose(-2, -1) * scale
mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
attn = torch.softmax(scores, dim=-1)
expected = attn @ V
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff {(out - expected).abs().max()}'
""",
        },
    ],
    "solution": '''def causal_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = scores.size(-1)
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)''',
    "demo": """torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = causal_attention(Q, K, V)
print("Pos 0 == V[0]?", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))""",

}