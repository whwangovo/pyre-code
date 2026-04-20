"""RMSNorm implementation task."""

TASK = {
    "title": "Implement RMSNorm",
    "title_zh": "实现 RMSNorm",
    "difficulty": "Medium",
    "description_en": "Implement RMSNorm (Root Mean Square Layer Normalization).\n\nRMSNorm is a simpler alternative to LayerNorm that skips mean subtraction, normalizing only by the root mean square of activations.\n\n**Signature:** `rms_norm(x, weight, eps=1e-6) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor (..., D)\n- `weight` — learnable scale parameter (D,)\n- `eps` — epsilon for numerical stability\n\n**Returns:** normalized tensor, same shape as x\n\n**Constraints:**\n- `RMS(x) = sqrt(mean(x^2) + eps)` over last dim\n- Output: `x / RMS(x) * weight`",
    "description_zh": "实现 RMSNorm（均方根层归一化）。\n\nRMSNorm 是 LayerNorm 的简化替代，跳过均值减法，仅通过激活值的均方根进行归一化。\n\n**签名:** `rms_norm(x, weight, eps=1e-6) -> Tensor`\n\n**参数:**\n- `x` — 输入张量 (..., D)\n- `weight` — 可学习的缩放参数 (D,)\n- `eps` — 数值稳定性的 epsilon\n\n**返回:** 归一化后的张量，形状与 x 相同\n\n**约束:**\n- `RMS(x) = sqrt(mean(x^2) + eps)` 沿最后一维\n- 输出：`x / RMS(x) * weight`",
    "function_name": "rms_norm",
    "hint": "`rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`\n`return x / rms * weight`  ← no mean subtraction",
    "hint_zh": "`rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`\n`return x / rms * weight`  ← 无需减均值",
    "tests": [
        {
            "name": "Basic behavior",
            "code": """
import torch
x = torch.randn(2, 8)
weight = torch.ones(8)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch'
""",
        },
        {
            "name": "With learned weight",
            "code": """
import torch
x = torch.randn(4, 16)
weight = torch.randn(16)
out = {fn}(x, weight)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch with non-trivial weight'
""",
        },
        {
            "name": "3-D input",
            "code": """
import torch
x = torch.randn(2, 4, 32)
weight = torch.ones(32)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch on 3-D: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch on 3-D'
""",
        },
        {
            "name": "Differs from LayerNorm (no mean subtraction)",
            "code": """
import torch
torch.manual_seed(0)
x = torch.ones(2, 8) * 5.0 + torch.randn(2, 8) * 0.1
weight = torch.ones(8)
out = {fn}(x, weight)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Should match RMS formula (no mean subtraction)'
# LayerNorm would subtract mean first, giving very different results
ln_out = torch.nn.functional.layer_norm(x, [8], weight, torch.zeros(8))
assert not torch.allclose(out, ln_out, atol=0.1), 'Output should differ from LayerNorm'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
x = torch.randn(2, 8, requires_grad=True)
weight = torch.ones(8, requires_grad=True)
out = {fn}(x, weight)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert weight.grad is not None, 'weight.grad is None'
""",
        },
    ],
    "solution": '''def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight''',
    "demo": """x = torch.randn(2, 8)
out = rms_norm(x, torch.ones(8))
print('RMS of output:', out.pow(2).mean(dim=-1).sqrt())""",

}