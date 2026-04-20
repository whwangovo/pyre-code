"""Rotary Position Embedding (RoPE) task."""

TASK = {
    "title": "Rotary Position Embedding (RoPE)",
    "title_zh": "旋转位置编码（RoPE）",
    "difficulty": "Medium",
    "description_en": "Implement Rotary Position Embedding (RoPE).\n\nRoPE encodes position information by rotating query and key vectors in pairs, enabling relative position awareness through dot-product properties.\n\n**Signature:** `apply_rope(q, k) -> (Tensor, Tensor)`\n\n**Parameters:**\n- `q` — query tensor (B, S, D)\n- `k` — key tensor (B, S, D)\n\n**Returns:** tuple of rotated (q, k) with same shapes\n\n**Constraints:**\n- Split into even/odd pairs, apply rotation with `angles = pos * 1/(10000^(2i/D))`\n- Must preserve vector norms\n- Dot products should depend only on relative position",
    "description_zh": "实现旋转位置编码（RoPE）。\n\nRoPE 通过成对旋转查询和键向量来编码位置信息，利用点积性质实现相对位置感知。\n\n**签名:** `apply_rope(q, k) -> (Tensor, Tensor)`\n\n**参数:**\n- `q` — 查询张量 (B, S, D)\n- `k` — 键张量 (B, S, D)\n\n**返回:** 旋转后的 (q, k) 元组，形状不变\n\n**约束:**\n- 按奇偶对分割，使用 `angles = pos * 1/(10000^(2i/D))` 旋转\n- 必须保持向量范数不变\n- 点积应仅依赖于相对位置",
    "function_name": "apply_rope",
    "hint": "1. `freqs = 1 / 10000^(2i/D)` for i in `range(D//2)`\n2. `angles = pos[:, None] * freqs` → `cos_a`, `sin_a`\n3. Split: `x_e = x[..., 0::2]`, `x_o = x[..., 1::2]`\n4. Rotate: `[x_e*cos - x_o*sin, x_e*sin + x_o*cos]` → stack → flatten",
    "hint_zh": "1. `freqs = 1 / 10000^(2i/D)`，i 取 `range(D//2)`\n2. `angles = pos[:, None] * freqs` → `cos_a`、`sin_a`\n3. 拆分：`x_e = x[..., 0::2]`，`x_o = x[..., 1::2]`\n4. 旋转：`[x_e*cos - x_o*sin, x_e*sin + x_o*cos]` → stack → flatten",
    "tests": [
        {
            "name": "Output shapes",
            "code": "\nimport torch\nq = torch.randn(2, 8, 64)\nk = torch.randn(2, 8, 64)\nq_rot, k_rot = {fn}(q, k)\nassert q_rot.shape == q.shape, f'Q shape: {q_rot.shape}'\nassert k_rot.shape == k.shape, f'K shape: {k_rot.shape}'\n"
        },
        {
            "name": "Preserves norm",
            "code": "\nimport torch\ntorch.manual_seed(0)\nq = torch.randn(1, 16, 32)\nk = torch.randn(1, 16, 32)\nq_rot, k_rot = {fn}(q, k)\nassert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4), 'RoPE should preserve norms'\n"
        },
        {
            "name": "Relative position property",
            "code": "\nimport torch\ntorch.manual_seed(0)\nq = torch.randn(1, 8, 16)\nk = torch.randn(1, 8, 16)\nq_rot, k_rot = {fn}(q, k)\nq2 = torch.cat([torch.zeros(1, 3, 16), q], dim=1)\nk2 = torch.cat([torch.zeros(1, 3, 16), k], dim=1)\nq2_rot, k2_rot = {fn}(q2, k2)\n# Same relative distance of 2: pos 0 vs pos 2, and pos 3 vs pos 5\ndot1 = (q_rot[:, 0] * k_rot[:, 2]).sum(dim=-1)\ndot2 = (q2_rot[:, 3] * k2_rot[:, 5]).sum(dim=-1)\nassert torch.allclose(dot1, dot2, atol=1e-4), 'Dot product should depend on relative position only'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nq = torch.randn(1, 4, 8, requires_grad=True)\nk = torch.randn(1, 4, 8, requires_grad=True)\nqr, kr = {fn}(q, k)\n(qr.sum() + kr.sum()).backward()\nassert q.grad is not None and k.grad is not None, 'Missing gradients'\n"
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 2, 4
q = torch.randn(B, S, D)
k = torch.randn(B, S, D)
q_rot, k_rot = {fn}(q, k)
# Reference: for each position p and pair i, angle = p / 10000^(2i/D)
pos = torch.arange(S).float()
freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (S, D//2)
cos_a = torch.cos(angles)
sin_a = torch.sin(angles)
def rotate_ref(x):
    x_e = x[..., 0::2]
    x_o = x[..., 1::2]
    r_e = x_e * cos_a - x_o * sin_a
    r_o = x_e * sin_a + x_o * cos_a
    return torch.stack([r_e, r_o], dim=-1).flatten(-2)
expected_q = rotate_ref(q)
expected_k = rotate_ref(k)
assert torch.allclose(q_rot, expected_q, atol=1e-5), f'Q rotation mismatch: max diff {(q_rot - expected_q).abs().max()}'
assert torch.allclose(k_rot, expected_k, atol=1e-5), f'K rotation mismatch: max diff {(k_rot - expected_k).abs().max()}'
""",
        },
    ],
    "solution": '''def apply_rope(q, k):
    B, S, D = q.shape
    pos = torch.arange(S, device=q.device).unsqueeze(1).float()
    dim = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (10000.0 ** (dim / D))
    angles = pos * freqs
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    def rotate(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos_a - x2 * sin_a,
                            x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)

    return rotate(q), rotate(k)''',
    "demo": """q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
qr, kr = apply_rope(q, k)
print('Shape preserved:', qr.shape == q.shape)
print('Norm preserved:', torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))""",

}