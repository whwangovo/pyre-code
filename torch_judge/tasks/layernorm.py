"""LayerNorm implementation task."""

TASK = {
    "title": "Implement LayerNorm",
    "title_zh": "实现 LayerNorm",
    "difficulty": "Medium",
    "description_en": "Implement Layer Normalization.\n\nLayerNorm normalizes each sample across the feature dimension, stabilizing training without dependence on batch size.\n\n**Signature:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor (..., D)\n- `gamma` — scale parameter (D,)\n- `beta` — shift parameter (D,)\n- `eps` — epsilon for numerical stability\n\n**Returns:** normalized tensor, same shape as x\n\n**Constraints:**\n- Normalize over the last dimension\n- Use `unbiased=False` for variance\n- Must match `F.layer_norm`",
    "description_zh": "实现层归一化。\n\nLayerNorm 对每个样本沿特征维度进行归一化，不依赖批大小即可稳定训练。\n\n**签名:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`\n\n**参数:**\n- `x` — 输入张量 (..., D)\n- `gamma` — 缩放参数 (D,)\n- `beta` — 偏移参数 (D,)\n- `eps` — 数值稳定性的 epsilon\n\n**返回:** 归一化后的张量，形状与 x 相同\n\n**约束:**\n- 沿最后一个维度归一化\n- 方差使用 `unbiased=False`\n- 必须与 `F.layer_norm` 一致",
    "function_name": "my_layer_norm",
    "hint": "1. `mean = x.mean(dim=-1, keepdim=True)`\n2. `var = x.var(dim=-1, keepdim=True, unbiased=False)`\n3. `x_norm = (x - mean) / sqrt(var + eps)` → `gamma * x_norm + beta`",
    "hint_zh": "1. `mean = x.mean(dim=-1, keepdim=True)`\n2. `var = x.var(dim=-1, keepdim=True, unbiased=False)`\n3. `x_norm = (x - mean) / sqrt(var + eps)` → `gamma * x_norm + beta`",
    "tests": [
        {
            "name": "Shape and basic behavior",
            "code": """
import torch
x = torch.randn(2, 3, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = {fn}(x, gamma, beta)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch vs F.layer_norm'
""",
        },
        {
            "name": "With learned parameters",
            "code": """
import torch
x = torch.randn(4, 16)
gamma = torch.randn(16)
beta = torch.randn(16)
out = {fn}(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [16], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch with non-trivial gamma/beta'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
x = torch.randn(2, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
out = {fn}(x, gamma, beta)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
""",
        },
    ],
    "solution": '''def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta''',
    "demo": """x = torch.randn(2, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = my_layer_norm(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
print("Match ref?", torch.allclose(out, ref, atol=1e-4))""",

}