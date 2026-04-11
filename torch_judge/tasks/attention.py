"""Softmax Attention task."""

TASK = {
    "title": "Softmax Attention",
    "difficulty": "Easy",
    "description_en": "Implement scaled dot-product attention.\n\nThis is the core attention mechanism: compute similarity scores between queries and keys, then use them to weight values.\n\n**Signature:** `scaled_dot_product_attention(Q, K, V) -> Tensor`\n\n**Parameters:**\n- `Q` — query tensor (B, S_q, D)\n- `K` — key tensor (B, S_k, D)\n- `V` — value tensor (B, S_k, D_v)\n\n**Returns:** weighted values tensor (B, S_q, D_v)\n\n**Constraints:**\n- Scale scores by `1/sqrt(d_k)`\n- Support cross-attention (S_q != S_k)",
    "description_zh": "实现缩放点积注意力。\n\n这是核心注意力机制：计算查询和键之间的相似度分数，然后用它们对值进行加权。\n\n**签名:** `scaled_dot_product_attention(Q, K, V) -> Tensor`\n\n**参数:**\n- `Q` — 查询张量 (B, S_q, D)\n- `K` — 键张量 (B, S_k, D)\n- `V` — 值张量 (B, S_k, D_v)\n\n**返回:** 加权后的值张量 (B, S_q, D_v)\n\n**约束:**\n- 分数需除以 `sqrt(d_k)` 进行缩放\n- 支持交叉注意力（S_q != S_k）",
    "function_name": "scaled_dot_product_attention",
    "hint": "`scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(d_k)` → `torch.bmm(softmax(scores, dim=-1), V)`.",
    "hint_zh": "`scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(d_k)` → `torch.bmm(softmax(scores, dim=-1), V)`。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'
""",
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
weights = torch.softmax(scores, dim=-1)
ref = torch.bmm(weights, V)
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch vs reference'
""",
        },
        {
            "name": "Gradient check",
            "code": """
import torch, math
torch.manual_seed(42)
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None, 'Q.grad is None'
assert K.grad is not None, 'K.grad is None'
assert V.grad is not None, 'V.grad is None'
""",
        },
        {
            "name": "Cross-attention (seq_q != seq_k)",
            "code": """
import torch
Q = torch.randn(1, 3, 16)
K = torch.randn(1, 5, 16)
V = torch.randn(1, 5, 32)
out = {fn}(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'
""",
        },
    ],
    "solution": '''def scaled_dot_product_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)''',
}
