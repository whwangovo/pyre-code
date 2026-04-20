"""Paged Attention task."""

TASK = {
    "title": "Paged Attention",
    "title_zh": "分页注意力",
    "difficulty": "Hard",
    "description_en": "Implement paged attention (vLLM-style).\n\nInstead of a contiguous KV cache, keys and values are stored in fixed-size pages (blocks). A block table maps logical block indices to physical page indices, enabling non-contiguous memory allocation.\n\n**Signature:** `paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size) -> Tensor`\n\n**Parameters:**\n- `Q` — query tensor `(B, 1, D)` (single decode step)\n- `k_pages` — key pages `(num_pages, block_size, D)`\n- `v_pages` — value pages `(num_pages, block_size, D)`\n- `block_table` — integer tensor `(B, max_blocks)` mapping logical → physical page\n- `context_len` — number of valid KV tokens per sequence (scalar)\n- `block_size` — tokens per page\n\n**Returns:** output tensor `(B, 1, D)`\n\n**Steps:** Use block_table to gather K/V pages, slice to context_len, then compute standard scaled dot-product attention.",
    "description_zh": "实现分页注意力（vLLM 风格）。\n\nKV 缓存不再连续存储，而是存在固定大小的页（block）中。block table 将逻辑块索引映射到物理页索引，实现非连续内存分配。\n\n**签名:** `paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size) -> Tensor`\n\n**参数:**\n- `Q` — 查询张量 `(B, 1, D)`（单步解码）\n- `k_pages` — 键页 `(num_pages, block_size, D)`\n- `v_pages` — 值页 `(num_pages, block_size, D)`\n- `block_table` — 整数张量 `(B, max_blocks)`，逻辑块 → 物理页\n- `context_len` — 每条序列有效 KV token 数（标量）\n- `block_size` — 每页 token 数\n\n**返回:** 输出张量 `(B, 1, D)`\n\n**步骤:** 用 block_table gather K/V 页，截取到 context_len，再做标准缩放点积注意力。",
    "function_name": "paged_attention",
    "hint": "for b in range(B):\n  K = k_pages[block_table[b]].reshape(-1, D)[:context_len]  # gather + slice\n  V = v_pages[block_table[b]].reshape(-1, D)[:context_len]\n  scores = Q[b] @ K.T / √D → softmax → @ V",
    "hint_zh": "for b in range(B):\n  K = k_pages[block_table[b]].reshape(-1, D)[:context_len]  # gather + 截取\n  V = v_pages[block_table[b]].reshape(-1, D)[:context_len]\n  scores = Q[b] @ K.T / √D → softmax → @ V",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
B, D, block_size, num_pages = 2, 16, 4, 8
context_len = 6
max_blocks = (context_len + block_size - 1) // block_size
Q = torch.randn(B, 1, D)
k_pages = torch.randn(num_pages, block_size, D)
v_pages = torch.randn(num_pages, block_size, D)
block_table = torch.zeros(B, max_blocks, dtype=torch.long)
for b in range(B):
    for i in range(max_blocks):
        block_table[b, i] = b * max_blocks + i
out = {fn}(Q, k_pages, v_pages, block_table, context_len, block_size)
assert out.shape == (B, 1, D), f'Expected ({B}, 1, {D}), got {out.shape}'
""",
        },
        {
            "name": "Matches standard attention with contiguous layout",
            "code": """
import torch
torch.manual_seed(1)
B, S, D, block_size = 1, 8, 16, 4
num_blocks = S // block_size
# Build contiguous K/V as pages
K_full = torch.randn(B, S, D)
V_full = torch.randn(B, S, D)
Q = torch.randn(B, 1, D)
# Pages: (num_blocks, block_size, D)
k_pages = K_full[0].view(num_blocks, block_size, D)
v_pages = V_full[0].view(num_blocks, block_size, D)
# Identity block table: logical block i -> physical page i
block_table = torch.arange(num_blocks).unsqueeze(0)  # (1, num_blocks)
out_paged = {fn}(Q, k_pages, v_pages, block_table, S, block_size)
# Standard attention
scale = D ** -0.5
scores = (Q @ K_full.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
out_std = attn @ V_full
assert torch.allclose(out_paged, out_std, atol=1e-5), f'Paged attention should match standard attention'
""",
        },
        {
            "name": "Works with non-contiguous block table",
            "code": """
import torch
torch.manual_seed(2)
B, S, D, block_size = 1, 8, 16, 4
num_blocks = S // block_size  # 2 blocks
K_full = torch.randn(B, S, D)
V_full = torch.randn(B, S, D)
Q = torch.randn(B, 1, D)
# Allocate 4 physical pages, use pages 3 and 1 (non-contiguous)
num_phys_pages = 4
k_pages = torch.zeros(num_phys_pages, block_size, D)
v_pages = torch.zeros(num_phys_pages, block_size, D)
phys_indices = [3, 1]  # logical block 0 -> page 3, logical block 1 -> page 1
for logical, phys in enumerate(phys_indices):
    k_pages[phys] = K_full[0, logical*block_size:(logical+1)*block_size]
    v_pages[phys] = V_full[0, logical*block_size:(logical+1)*block_size]
block_table = torch.tensor(phys_indices).unsqueeze(0)  # (1, 2)
out_paged = {fn}(Q, k_pages, v_pages, block_table, S, block_size)
# Standard attention
scale = D ** -0.5
scores = (Q @ K_full.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
out_std = attn @ V_full
assert torch.allclose(out_paged, out_std, atol=1e-5), 'Non-contiguous block table should give same result'
""",
        },
        {
            "name": "Gradient flows",
            "code": """
import torch
B, D, block_size = 1, 8, 4
Q = torch.randn(B, 1, D, requires_grad=True)
k_pages = torch.randn(4, block_size, D, requires_grad=True)
v_pages = torch.randn(4, block_size, D, requires_grad=True)
block_table = torch.arange(4).unsqueeze(0)
out = {fn}(Q, k_pages, v_pages, block_table, 16, block_size)
out.sum().backward()
assert Q.grad is not None and k_pages.grad is not None
""",
        },
    ],
    "solution": '''def paged_attention(Q, k_pages, v_pages, block_table, context_len, block_size):
    B, _, D = Q.shape
    outputs = []
    for b in range(B):
        # Gather K/V pages for this sequence
        phys_blocks = block_table[b]  # (max_blocks,)
        K_gathered = k_pages[phys_blocks].reshape(-1, D)  # (max_blocks*block_size, D)
        V_gathered = v_pages[phys_blocks].reshape(-1, D)
        # Slice to actual context length
        K_ctx = K_gathered[:context_len].unsqueeze(0)  # (1, context_len, D)
        V_ctx = V_gathered[:context_len].unsqueeze(0)
        # Scaled dot-product attention
        scale = D ** -0.5
        scores = (Q[b:b+1] @ K_ctx.transpose(-2, -1)) * scale  # (1, 1, context_len)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V_ctx  # (1, 1, D)
        outputs.append(out)
    return torch.cat(outputs, dim=0)''',
    "demo": """torch.manual_seed(0)
B, S, D = 2, 8, 16
block_size = 4
num_blocks = S // block_size

K_full = torch.randn(B, S, D)
V_full = torch.randn(B, S, D)
Q = torch.randn(B, 1, D)

scale = D ** -0.5
scores_ref = (Q @ K_full.transpose(-2, -1)) * scale
ref_out = torch.softmax(scores_ref, dim=-1) @ V_full

total_pages = B * num_blocks
k_pages = torch.zeros(total_pages, block_size, D)
v_pages = torch.zeros(total_pages, block_size, D)
block_table = []
for b in range(B):
    page_ids = []
    for blk in range(num_blocks):
        pid = b * num_blocks + blk
        k_pages[pid] = K_full[b, blk*block_size:(blk+1)*block_size]
        v_pages[pid] = V_full[b, blk*block_size:(blk+1)*block_size]
        page_ids.append(pid)
    block_table.append(page_ids)

paged_out = paged_attention(Q, k_pages, v_pages, block_table, context_len=S, block_size=block_size)

print('Shape:', paged_out.shape)
print('Max diff vs reference:', (paged_out - ref_out).abs().max().item())
print('Match:', torch.allclose(paged_out, ref_out, atol=1e-5))""",

}
