"""Differential Attention task."""

TASK = {
    "title": "Differential Attention",
    "title_zh": "差分注意力",
    "difficulty": "Hard",
    "description_en": "Implement Differential Attention from the ICLR 2025 paper \"Differential Transformer\".\n\nThe key idea: split Q and K each into two halves, compute two separate softmax attention maps, then subtract them (scaled by a learnable lambda) to cancel noise and improve focus on relevant context.\n\n**Signature:** `diff_attention(Q, K, V, lambda_val) -> Tensor`\n\n**Parameters:**\n- `Q` — query tensor (B, S, 2*D_h)\n- `K` — key tensor (B, S, 2*D_h)\n- `V` — value tensor (B, S, D_v)\n- `lambda_val` — scalar float or 0-dim tensor controlling noise cancellation\n\n**Returns:** output tensor (B, S, D_v)\n\n**Formula:**\n```\nout = (softmax(Q1 @ K1.T / √D_h) - lambda_val * softmax(Q2 @ K2.T / √D_h)) @ V\n```\n\n**Constraints:**\n- Split Q into Q1, Q2 along last dim; same for K\n- Use `torch.softmax(..., dim=-1)` (not F.softmax)\n- Scale by 1/√D_h before softmax",
    "description_zh": "实现 ICLR 2025 论文《Differential Transformer》中的差分注意力机制。\n\n核心思想：将 Q 和 K 各自分成两半，分别计算两个 softmax 注意力图，然后相减（乘以可学习的 lambda）以消除噪声，提升对相关上下文的聚焦能力。\n\n**签名:** `diff_attention(Q, K, V, lambda_val) -> Tensor`\n\n**参数:**\n- `Q` — 查询张量 (B, S, 2*D_h)\n- `K` — 键张量 (B, S, 2*D_h)\n- `V` — 值张量 (B, S, D_v)\n- `lambda_val` — 标量浮点数或 0 维张量，控制噪声消除强度\n\n**返回:** 输出张量 (B, S, D_v)\n\n**公式:**\n```\nout = (softmax(Q1 @ K1.T / √D_h) - lambda_val * softmax(Q2 @ K2.T / √D_h)) @ V\n```\n\n**约束:**\n- 沿最后一维将 Q 拆分为 Q1、Q2；K 同理\n- 使用 `torch.softmax(..., dim=-1)`（不能用 F.softmax）\n- softmax 前除以 √D_h",
    "function_name": "diff_attention",
    "hint": "1. `Q1,Q2 = Q[...,:D_h], Q[...,D_h:]`; same for K\n2. `scale = D_h**-0.5`\n3. `A1 = softmax(Q1@K1.T * scale)`, `A2 = softmax(Q2@K2.T * scale)`\n4. `return (A1 - lambda_val*A2) @ V`",
    "hint_zh": "1. `Q1,Q2 = Q[...,:D_h], Q[...,D_h:]`；K 同理\n2. `scale = D_h**-0.5`\n3. `A1 = softmax(Q1@K1.T * scale)`，`A2 = softmax(Q2@K2.T * scale)`\n4. `return (A1 - lambda_val*A2) @ V`",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
B, S, D_h, D_v = 2, 6, 16, 32
Q = torch.randn(B, S, 2*D_h)
K = torch.randn(B, S, 2*D_h)
V = torch.randn(B, S, D_v)
out = {fn}(Q, K, V, 0.5)
assert out.shape == (B, S, D_v), f'Expected ({B}, {S}, {D_v}), got {out.shape}'
"""
        },
        {
            "name": "lambda=0 reduces to standard attention",
            "code": """
import torch
torch.manual_seed(1)
B, S, D_h, D_v = 1, 4, 8, 16
Q = torch.randn(B, S, 2*D_h)
K = torch.randn(B, S, 2*D_h)
V = torch.randn(B, S, D_v)
out = {fn}(Q, K, V, 0.0)
# Manually compute standard attention with Q1, K1
Q1 = Q[..., :D_h]
K1 = K[..., :D_h]
scale = D_h ** -0.5
A = torch.softmax(Q1 @ K1.transpose(-2, -1) * scale, dim=-1)
expected = A @ V
assert torch.allclose(out, expected, atol=1e-5), 'lambda=0 should match standard attention on Q1/K1'
"""
        },
        {
            "name": "lambda=1 with identical halves gives zero",
            "code": """
import torch
torch.manual_seed(2)
B, S, D_h, D_v = 1, 5, 8, 12
# Make Q and K such that Q1==Q2 and K1==K2
half_Q = torch.randn(B, S, D_h)
half_K = torch.randn(B, S, D_h)
Q = torch.cat([half_Q, half_Q], dim=-1)
K = torch.cat([half_K, half_K], dim=-1)
V = torch.randn(B, S, D_v)
out = {fn}(Q, K, V, 1.0)
assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), 'Identical halves with lambda=1 should give zero output'
"""
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(3)
B, S, D_h, D_v = 1, 4, 8, 8
Q = torch.randn(B, S, 2*D_h, requires_grad=True)
K = torch.randn(B, S, 2*D_h, requires_grad=True)
V = torch.randn(B, S, D_v, requires_grad=True)
lam = torch.tensor(0.3, requires_grad=True)
out = {fn}(Q, K, V, lam)
out.sum().backward()
assert Q.grad is not None, 'Missing gradient for Q'
assert K.grad is not None, 'Missing gradient for K'
assert V.grad is not None, 'Missing gradient for V'
assert lam.grad is not None, 'Missing gradient for lambda_val'
"""
        }
    ],
    "solution": '''def diff_attention(Q, K, V, lambda_val):
    B, S, D2 = Q.shape
    D_h = D2 // 2
    Q1, Q2 = Q[..., :D_h], Q[..., D_h:]
    K1, K2 = K[..., :D_h], K[..., D_h:]
    scale = D_h ** -0.5
    A1 = torch.softmax(Q1 @ K1.transpose(-2, -1) * scale, dim=-1)
    A2 = torch.softmax(Q2 @ K2.transpose(-2, -1) * scale, dim=-1)
    return (A1 - lambda_val * A2) @ V''',
    "demo": """torch.manual_seed(0)
B, S, D2, D_v = 2, 4, 8, 6
Q = torch.randn(B, S, D2)
K = torch.randn(B, S, D2)
V = torch.randn(B, S, D_v)

D_h = D2 // 2
scale = D_h ** -0.5
standard = torch.softmax(Q[..., :D_h] @ K[..., :D_h].transpose(-2, -1) * scale, dim=-1) @ V
diff_zero = diff_attention(Q, K, V, lambda_val=0.0)
print("lambda=0 matches standard attention:", torch.allclose(diff_zero, standard, atol=1e-6))

Q_same = torch.cat([Q[..., :D_h], Q[..., :D_h]], dim=-1)
K_same = torch.cat([K[..., :D_h], K[..., :D_h]], dim=-1)
diff_one = diff_attention(Q_same, K_same, V, lambda_val=1.0)
print("lambda=1 with identical halves gives zero:", torch.allclose(diff_one, torch.zeros_like(diff_one), atol=1e-6))
print("Output shape:", diff_zero.shape)  # (2, 4, 6)""",

}
