"""NTK-aware RoPE Scaling task."""

TASK = {
    "title": "NTK-aware RoPE Scaling",
    "title_zh": "NTK-aware RoPE 缩放",
    "difficulty": "Easy",
    "description_en": "Implement NTK-aware RoPE scaling for long-context extrapolation.\n\nStandard RoPE degrades on sequences longer than the training context. NTK-aware scaling adjusts the base frequency so that high-frequency dimensions are preserved while low-frequency ones are stretched, enabling extrapolation without fine-tuning.\n\n**Signature:** `ntk_rope(q, k, scale) -> (Tensor, Tensor)`\n\n**Parameters:**\n- `q, k` — tensors of shape `(B, S, D)`\n- `scale` — context length ratio (new_len / train_len), e.g. 4.0 for 4× context\n\n**Returns:** rotated `(q, k)` with same shapes\n\n**NTK base:** `base_new = 10000 * scale^(D/(D-2))`\n\nThen apply standard RoPE with the new base.",
    "description_zh": "实现 NTK-aware RoPE 缩放，用于长上下文外推。\n\n标准 RoPE 在超过训练上下文长度的序列上性能下降。NTK-aware 缩放调整基础频率，保留高频维度同时拉伸低频维度，无需微调即可外推。\n\n**签名:** `ntk_rope(q, k, scale) -> (Tensor, Tensor)`\n\n**参数:**\n- `q, k` — 形状 `(B, S, D)` 的张量\n- `scale` — 上下文长度比例（新长度/训练长度），如 4× 上下文传 4.0\n\n**返回:** 旋转后的 `(q, k)`，形状不变\n\n**NTK 基底:** `base_new = 10000 * scale^(D/(D-2))`\n\n然后用新基底应用标准 RoPE。",
    "function_name": "ntk_rope",
    "hint": "`new_base = 10000 * scale**(D/(D-2))` → apply standard RoPE with `new_base` instead of `10000`.",
    "hint_zh": "`new_base = 10000 * scale**(D/(D-2))` → 用 `new_base` 替换 `10000` 应用标准 RoPE。",
    "tests": [
        {
            "name": "Output shapes",
            "code": """
import torch
torch.manual_seed(0)
q = torch.randn(2, 16, 32)
k = torch.randn(2, 16, 32)
q_rot, k_rot = {fn}(q, k, scale=4.0)
assert q_rot.shape == q.shape, f'Q shape mismatch: {q_rot.shape}'
assert k_rot.shape == k.shape, f'K shape mismatch: {k_rot.shape}'
""",
        },
        {
            "name": "scale=1 matches standard RoPE",
            "code": """
import torch
torch.manual_seed(1)
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
q_ntk, k_ntk = {fn}(q, k, scale=1.0)
# Standard RoPE with base=10000
B, S, D = q.shape
pos = torch.arange(S).float().unsqueeze(1)
dim = torch.arange(0, D, 2).float()
freqs = 1.0 / (10000.0 ** (dim / D))
angles = pos * freqs
cos_a, sin_a = torch.cos(angles), torch.sin(angles)
def rotate(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*cos_a - x2*sin_a, x1*sin_a + x2*cos_a], dim=-1).flatten(-2)
q_std, k_std = rotate(q), rotate(k)
assert torch.allclose(q_ntk, q_std, atol=1e-5), 'scale=1 should match standard RoPE'
""",
        },
        {
            "name": "Preserves norms",
            "code": """
import torch
torch.manual_seed(2)
q = torch.randn(1, 12, 32)
k = torch.randn(1, 12, 32)
q_rot, k_rot = {fn}(q, k, scale=8.0)
assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4), 'RoPE must preserve norms'
""",
        },
        {
            "name": "scale>1 changes frequencies",
            "code": """
import torch
torch.manual_seed(3)
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
q1, _ = {fn}(q, k, scale=1.0)
q4, _ = {fn}(q, k, scale=4.0)
assert not torch.allclose(q1, q4, atol=1e-3), 'Different scales should produce different rotations'
""",
        },
        {
            "name": "NTK base formula correctness",
            "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 4, 8
scale = 4.0
q = torch.randn(B, S, D)
k = torch.randn(B, S, D)
q_rot, k_rot = {fn}(q, k, scale)
# Reference with exact NTK base
new_base = 10000.0 * (scale ** (D / (D - 2)))
pos = torch.arange(S).float().unsqueeze(1)
dim = torch.arange(0, D, 2).float()
freqs = 1.0 / (new_base ** (dim / D))
angles = pos * freqs
cos_a, sin_a = torch.cos(angles), torch.sin(angles)
def rotate(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*cos_a - x2*sin_a, x1*sin_a + x2*cos_a], dim=-1).flatten(-2)
assert torch.allclose(q_rot, rotate(q), atol=1e-5), 'NTK base formula mismatch'
""",
        },
    ],
    "solution": '''def ntk_rope(q, k, scale):
    B, S, D = q.shape
    new_base = 10000.0 * (scale ** (D / (D - 2)))
    pos = torch.arange(S, device=q.device).float().unsqueeze(1)
    dim = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (new_base ** (dim / D))
    angles = pos * freqs
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    def rotate(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos_a - x2 * sin_a,
                            x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)

    return rotate(q), rotate(k)''',
    "demo": """B, S, D = 1, 8, 16
q = torch.randn(B, S, D)
k = torch.randn(B, S, D)

q1, k1 = ntk_rope(q, k, scale=1.0)
q4, k4 = ntk_rope(q, k, scale=4.0)

print("scale=1 base:", 10000.0 * (1.0 ** (D / (D - 2))))
print("scale=4 base:", 10000.0 * (4.0 ** (D / (D - 2))))
print()
print("Norm preservation (scale=1):")
print("  q input norm:", q.norm(dim=-1).mean().item())
print("  q output norm:", q1.norm(dim=-1).mean().item())
print()
print("Norm preservation (scale=4):")
print("  q input norm:", q.norm(dim=-1).mean().item())
print("  q output norm:", q4.norm(dim=-1).mean().item())""",

}
