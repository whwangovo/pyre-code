"""Grouped Query Attention task."""

TASK = {
    "title": "Grouped Query Attention",
    "title_zh": "分组查询注意力（GQA）",
    "difficulty": "Hard",
    "description_en": "Implement Grouped Query Attention (GQA).\n\nGQA uses fewer KV heads than query heads, sharing each KV head across a group of query heads. This reduces KV cache size while preserving quality.\n\n**Signature:** `GroupQueryAttention(d_model, num_heads, num_kv_heads)`\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (B, S, d_model)\n\n**Returns:** attention output (B, S, d_model)\n\n**Constraints:**\n- W_k/W_v project to `num_kv_heads * d_k` dimensions\n- Expand KV heads with `repeat_interleave` to match query heads\n- Degenerates to standard MHA when `num_kv_heads == num_heads`",
    "description_zh": "实现分组查询注意力（GQA）。\n\nGQA 使用比查询头更少的 KV 头，每个 KV 头在一组查询头之间共享，在保持质量的同时减少 KV 缓存大小。\n\n**签名:** `GroupQueryAttention(d_model, num_heads, num_kv_heads)`\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (B, S, d_model)\n\n**返回:** 注意力输出 (B, S, d_model)\n\n**约束:**\n- W_k/W_v 投影到 `num_kv_heads * d_k` 维\n- 使用 `repeat_interleave` 扩展 KV 头以匹配查询头数\n- 当 `num_kv_heads == num_heads` 时退化为标准 MHA",
    "function_name": "GroupQueryAttention",
    "hint": "1. `W_q` → `(B,H,S,d_k)`, `W_k`/`W_v` → `(B,KV,S,d_k)`\n2. `K = K.repeat_interleave(H//KV, dim=1)` to expand KV heads\n3. Scaled dot-product attn → reshape to `(B,S,d_model)`",
    "hint_zh": "1. `W_q` → `(B,H,S,d_k)`，`W_k`/`W_v` → `(B,KV,S,d_k)`\n2. `K = K.repeat_interleave(H//KV, dim=1)` 扩展 KV 头\n3. 缩放点积注意力 → reshape 为 `(B,S,d_model)`",
    "tests": [
        {
            "name": "Is nn.Module",
            "code": "\nimport torch, torch.nn as nn\ngqa = {fn}(d_model=16, num_heads=4, num_kv_heads=2)\nassert isinstance(gqa, nn.Module), 'GroupQueryAttention should inherit from nn.Module'\n",
        },
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
out = gqa.forward(torch.randn(2, 6, 32))
assert out.shape == (2, 6, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "nn.Linear with correct shapes",
            "code": """
import torch, torch.nn as nn
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
d_k = 32 // 8
assert isinstance(gqa.W_q, nn.Linear) and gqa.W_q.weight.shape == (32, 32), f'W_q wrong'
assert isinstance(gqa.W_k, nn.Linear) and gqa.W_k.weight.shape == (2 * d_k, 32), f'W_k shape: {gqa.W_k.weight.shape}'
assert isinstance(gqa.W_v, nn.Linear) and gqa.W_v.weight.shape == (2 * d_k, 32), f'W_v shape: {gqa.W_v.weight.shape}'
assert isinstance(gqa.W_o, nn.Linear), 'W_o should be nn.Linear'
""",
        },
        {
            "name": "Degenerates to MHA when kv_heads == heads",
            "code": """
import torch
torch.manual_seed(42)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=4)
out = gqa.forward(torch.randn(1, 4, 16))
assert out.shape == (1, 4, 16)
assert gqa.W_k.weight.shape == (16, 16), 'Full KV when kv_heads == heads'
""",
        },
        {
            "name": "KV heads are shared correctly",
            "code": """
import torch
torch.manual_seed(0)
D, H, KV = 16, 4, 2
d_k = D // H
gqa = {fn}(d_model=D, num_heads=H, num_kv_heads=KV)
x = torch.randn(1, 4, D)
k = gqa.W_k(x).view(1, 4, KV, d_k).transpose(1, 2)
k_exp = k.repeat_interleave(H // KV, dim=1)
assert torch.equal(k_exp[:, 0], k_exp[:, 1]), 'Heads 0,1 should share same K'
assert not torch.equal(k_exp[:, 0], k_exp[:, 2]), 'Different groups need different K'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(0)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
gqa.forward(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gqa.W_q.weight.grad is not None and gqa.W_k.weight.grad is not None, 'Missing weight gradients'
""",
        },
    ],
    "solution": '''class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)''',
    "demo": """gqa = GroupQueryAttention(32, 8, 2)
print('Output:', gqa.forward(torch.randn(1, 4, 32)).shape)""",

}