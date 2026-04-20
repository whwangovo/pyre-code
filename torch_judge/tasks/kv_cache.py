"""KV Cache Attention task."""

TASK = {
    "title": "KV Cache Attention",
    "title_zh": "KV Cache 注意力",
    "difficulty": "Hard",
    "description_en": "Implement multi-head attention with KV cache for efficient autoregressive decoding.\n\nKV caching stores previously computed keys and values, so each decode step only processes the new token instead of the full sequence.\n\n**Signature:** `KVCacheAttention(d_model, num_heads)` (nn.Module)\n\n**Forward:** `forward(x, cache=None) -> (Tensor, tuple)`\n- `x` — input tensor (B, S_new, d_model)\n- `cache` — optional (K, V) tuple from previous steps\n\n**Returns:** (output, (K_all, V_all)) where cache grows each step\n\n**Constraints:**\n- Concatenate new K/V with cached along sequence dim\n- Apply causal mask during prefill (S_new > 1)\n- Incremental decode must match full forward numerically",
    "description_zh": "实现带 KV 缓存的多头注意力，用于高效自回归解码。\n\nKV 缓存存储之前计算的键和值，使每个解码步骤只需处理新 token 而非完整序列。\n\n**签名:** `KVCacheAttention(d_model, num_heads)`（nn.Module）\n\n**前向传播:** `forward(x, cache=None) -> (Tensor, tuple)`\n- `x` — 输入张量 (B, S_new, d_model)\n- `cache` — 可选的前序步骤 (K, V) 元组\n\n**返回:** (输出, (K_all, V_all))，缓存逐步增长\n\n**约束:**\n- 将新 K/V 与缓存沿序列维度拼接\n- 预填充时（S_new > 1）应用因果掩码\n- 增量解码必须与完整前向传播数值一致",
    "function_name": "KVCacheAttention",
    "hint": "1. Project Q/K/V → reshape to `(B, num_heads, S, d_k)`\n2. If cache: K = cat([cache_K, K], dim=2),  V = cat([cache_V, V], dim=2)\n3. Apply causal mask when S_new > 1 (prefill)\n4. Scaled dot-product attention → output projection\n5. Return (output, (K_all, V_all))",
    "hint_zh": "1. 投影 Q/K/V → reshape 为 `(B, num_heads, S, d_k)`\n2. 有缓存时：K = cat([cache_K, K], dim=2),  V = cat([cache_V, V], dim=2)\n3. S_new > 1（预填充）时应用因果掩码\n4. 缩放点积注意力 → 输出投影\n5. 返回 (output, (K_all, V_all))",
    "tests": [
        {
            "name": "Output shape (no cache)",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
attn = {fn}(d_model=64, num_heads=4)
assert isinstance(attn, nn.Module), 'Must inherit from nn.Module'
x = torch.randn(2, 8, 64)
out, cache = attn(x)
assert out.shape == (2, 8, 64), f'Output shape: {out.shape}'
""",
        },
        {
            "name": "Cache structure",
            "code": """
import torch
torch.manual_seed(0)
attn = {fn}(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)
out, cache = attn(x)
assert isinstance(cache, tuple) and len(cache) == 2, 'Cache must be a (K, V) tuple'
k_cache, v_cache = cache
assert k_cache.shape == (2, 4, 8, 16), f'K cache shape: {k_cache.shape}, expected (2, 4, 8, 16)'
assert v_cache.shape == (2, 4, 8, 16), f'V cache shape: {v_cache.shape}, expected (2, 4, 8, 16)'
""",
        },
        {
            "name": "Decode step appends to cache",
            "code": """
import torch
torch.manual_seed(0)
attn = {fn}(d_model=32, num_heads=2)
x = torch.randn(1, 4, 32)
_, cache = attn(x)
new_token = torch.randn(1, 1, 32)
out, new_cache = attn(new_token, cache=cache)
assert out.shape == (1, 1, 32), f'Decode output shape: {out.shape}'
k_cache, v_cache = new_cache
assert k_cache.shape[2] == 5, f'Cache should grow: K has {k_cache.shape[2]} positions, expected 5'
assert v_cache.shape[2] == 5, f'Cache should grow: V has {v_cache.shape[2]} positions, expected 5'
""",
        },
        {
            "name": "Incremental decode matches full forward",
            "code": """
import torch
torch.manual_seed(42)
attn = {fn}(d_model=32, num_heads=2)
x = torch.randn(1, 6, 32)
full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)
assert torch.allclose(full_out, inc_out, atol=1e-5), 'Incremental decode must match full forward'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(0)
attn = {fn}(d_model=32, num_heads=2)
x = torch.randn(1, 4, 32, requires_grad=True)
out, cache = attn(x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in attn.parameters())
n_grad = sum(1 for p in attn.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
        },
    ],
    "solution": '''class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None):
        B, S_new, _ = x.shape

        q = self.W_q(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        new_cache = (k, v)
        S_total = k.shape[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if S_new > 1:
            # Causal mask for prefill: each query position can only attend to
            # positions up to itself in the full sequence
            S_past = S_total - S_new
            mask = torch.triu(
                torch.ones(S_new, S_total, device=x.device, dtype=torch.bool),
                diagonal=S_past + 1,
            )
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = self.W_o(attn.transpose(1, 2).contiguous().view(B, S_new, -1))
        return out, new_cache''',
    "demo": """torch.manual_seed(0)
attn = KVCacheAttention(d_model=64, num_heads=4)
x = torch.randn(1, 6, 64)

full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)

print('Full shape:', full_out.shape)
print('Match:', torch.allclose(full_out, inc_out, atol=1e-5))
print('Final cache K shape:', cache[0].shape)""",

}