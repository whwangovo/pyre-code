"""Sliding Window Attention task."""

TASK = {
    "title": "Sliding Window Attention",
    "title_zh": "滑动窗口注意力",
    "difficulty": "Medium",
    "description_en": "Implement sliding window attention.\n\nSliding window attention restricts each position to attend only within a fixed window, reducing complexity for long sequences while maintaining local context.\n\n**Signature:** `sliding_window_attention(Q, K, V, window_size) -> Tensor`\n\n**Parameters:**\n- `Q`, `K`, `V` — input tensors (B, S, D)\n- `window_size` — each position attends to positions within |i-j| <= window_size\n\n**Returns:** attention output (B, S, D)\n\n**Constraints:**\n- Mask positions where `|i - j| > window_size` with `-inf`\n- `window_size=0` means each position only sees itself\n- Large window equals full attention",
    "description_zh": "实现滑动窗口注意力。\n\n滑动窗口注意力限制每个位置只关注固定窗口内的位置，在保持局部上下文的同时降低长序列的复杂度。\n\n**签名:** `sliding_window_attention(Q, K, V, window_size) -> Tensor`\n\n**参数:**\n- `Q`, `K`, `V` — 输入张量 (B, S, D)\n- `window_size` — 每个位置关注 |i-j| <= window_size 范围内的位置\n\n**返回:** 注意力输出 (B, S, D)\n\n**约束:**\n- 用 `-inf` 掩盖 `|i - j| > window_size` 的位置\n- `window_size=0` 表示每个位置只能看到自身\n- 大窗口等同于全注意力",
    "function_name": "sliding_window_attention",
    "hint": "Standard attention + window mask. Build `|i-j|` matrix → mask positions where `|i-j| > window_size` with `-inf` → softmax → `@ V`.",
    "hint_zh": "标准注意力 + 窗口遮蔽。构造 `|i-j|` 矩阵 → 将 `|i-j| > window_size` 的位置填 `-inf` → softmax → `@ V`。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 16), window_size=2)
assert out.shape == (2, 8, 16), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "window_size=0 — only sees itself",
            "code": """
import torch
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V, window_size=0)
assert torch.allclose(out, V, atol=1e-5), 'window=0: each position should output V[i]'
""",
        },
        {
            "name": "Large window equals full attention",
            "code": """
import torch, math
torch.manual_seed(0)
B, S, D = 2, 6, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_win = {fn}(Q, K, V, window_size=S)
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
assert torch.allclose(out_win, ref, atol=1e-5), 'Large window should equal full attention'
""",
        },
        {
            "name": "Distant tokens don't affect output",
            "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 10, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V, window_size=1)
K2, V2 = K.clone(), V.clone()
K2[:, 5:] = torch.randn(B, 5, D)
V2[:, 5:] = torch.randn(B, 5, D)
out2 = {fn}(Q, K2, V2, window_size=1)
assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Distant tokens should not affect output'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
{fn}(Q, K, V, window_size=1).sum().backward()
assert Q.grad is not None, 'Q.grad is None'
""",
        },
    ],
    "solution": '''def sliding_window_attention(Q, K, V, window_size):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = Q.size(1)
    idx = torch.arange(S, device=Q.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)''',
    "demo": """Q=torch.randn(1,6,8); K=torch.randn(1,6,8); V=torch.randn(1,6,8)
print('window=0==V?', torch.allclose(sliding_window_attention(Q,K,V,0), V, atol=1e-5))""",

}