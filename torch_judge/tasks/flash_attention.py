"""Flash Attention (Tiled) task."""

TASK = {
    "title": "Flash Attention (Tiled)",
    "title_zh": "Flash Attention（分块）",
    "difficulty": "Hard",
    "description_en": "Implement tiled (flash) attention with online softmax.\n\nFlash attention processes Q/K/V in blocks to reduce memory usage, using the online softmax trick to maintain numerical correctness across tiles.\n\n**Signature:** `flash_attention(Q, K, V, block_size=32) -> Tensor`\n\n**Parameters:**\n- `Q`, `K`, `V` — input tensors (B, S, D)\n- `block_size` — tile size for blocking\n\n**Returns:** attention output (B, S, D), identical to standard attention\n\n**Constraints:**\n- Must match standard softmax attention numerically\n- Handle non-aligned sequence lengths (S not divisible by block_size)\n- Result must be invariant to block_size choice",
    "description_zh": "实现分块（Flash）注意力与在线 softmax。\n\nFlash 注意力将 Q/K/V 分块处理以减少内存使用，利用在线 softmax 技巧在分块间保持数值正确性。\n\n**签名:** `flash_attention(Q, K, V, block_size=32) -> Tensor`\n\n**参数:**\n- `Q`, `K`, `V` — 输入张量 (B, S, D)\n- `block_size` — 分块大小\n\n**返回:** 注意力输出 (B, S, D)，与标准注意力数值一致\n\n**约束:**\n- 必须与标准 softmax 注意力数值匹配\n- 处理非对齐序列长度（S 不能被 block_size 整除）\n- 结果不受 block_size 选择影响",
    "function_name": "flash_attention",
    "hint": (
        "Process Q in tiles. For each Q-block, iterate over K/V blocks:\n"
        "1. `block_max = scores.max(dim=-1)` → `new_max = max(row_max, block_max)`\n"
        "2. `correction = exp(row_max - new_max)` → rescale `acc` and `row_sum`\n"
        "3. `acc += exp(scores - new_max) @ V_block`\n"
        "4. `output = acc / row_sum`"
    ),
    "hint_zh": (
        "分块处理 Q。对每个 Q 块，遍历 K/V 块：\n"
        "1. `block_max = scores.max(dim=-1)` → `new_max = max(row_max, block_max)`\n"
        "2. `correction = exp(row_max - new_max)` → 重新缩放 `acc` 和 `row_sum`\n"
        "3. `acc += exp(scores - new_max) @ V_block`\n"
        "4. `output = acc / row_sum`"
    ),
    "tests": [
        {
            "name": "Matches standard attention",
            "code": "\nimport torch, math\ntorch.manual_seed(0)\nB, S, D = 2, 16, 8\nQ = torch.randn(B, S, D)\nK = torch.randn(B, S, D)\nV = torch.randn(B, S, D)\nout = {fn}(Q, K, V, block_size=4)\nscores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)\nref = torch.bmm(torch.softmax(scores, dim=-1), V)\nassert torch.allclose(out, ref, atol=1e-4), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "Non-aligned block size",
            "code": "\nimport torch, math\ntorch.manual_seed(42)\nB, S, D = 1, 7, 4\nQ, K, V = torch.randn(B,S,D), torch.randn(B,S,D), torch.randn(B,S,D)\nout = {fn}(Q, K, V, block_size=3)\nscores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)\nref = torch.bmm(torch.softmax(scores, dim=-1), V)\nassert torch.allclose(out, ref, atol=1e-4), 'Mismatch with non-aligned block size'\n"
        },
        {
            "name": "Block size invariant",
            "code": "\nimport torch\ntorch.manual_seed(0)\nQ, K, V = torch.randn(1,12,8), torch.randn(1,12,8), torch.randn(1,12,8)\nout4 = {fn}(Q, K, V, block_size=4)\nout6 = {fn}(Q, K, V, block_size=6)\nassert torch.allclose(out4, out6, atol=1e-4), 'Different block sizes should give same result'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nQ = torch.randn(1, 8, 4, requires_grad=True)\nK = torch.randn(1, 8, 4, requires_grad=True)\nV = torch.randn(1, 8, 4, requires_grad=True)\n{fn}(Q, K, V, block_size=4).sum().backward()\nassert Q.grad is not None, 'Q.grad is None'\n"
        }
    ],
    "solution": '''def flash_attention(Q, K, V, block_size=32):
    B, S, D = Q.shape
    output = torch.zeros_like(Q)
    for i in range(0, S, block_size):
        qi = Q[:, i:i+block_size]
        bs_q = qi.shape[1]
        row_max = torch.full((B, bs_q, 1), float('-inf'), device=Q.device)
        row_sum = torch.zeros(B, bs_q, 1, device=Q.device)
        acc = torch.zeros(B, bs_q, D, device=Q.device)
        for j in range(0, S, block_size):
            kj = K[:, j:j+block_size]
            vj = V[:, j:j+block_size]
            scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, block_max)
            correction = torch.exp(row_max - new_max)
            exp_scores = torch.exp(scores - new_max)
            acc = acc * correction + torch.bmm(exp_scores, vj)
            row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
            row_max = new_max
        output[:, i:i+block_size] = acc / row_sum
    return output''',
    "demo": """Q, K, V = torch.randn(1, 16, 8), torch.randn(1, 16, 8), torch.randn(1, 16, 8)
out = flash_attention(Q, K, V, block_size=4)
scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(8)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
print('Shape:', out.shape)
print('Max diff:', (out - ref).abs().max().item())""",

}