"""Softmax implementation task."""

TASK = {
    "title": "Implement Softmax",
    "title_zh": "实现 Softmax",
    "difficulty": "Easy",
    "description_en": "Implement the softmax function.\n\nSoftmax converts raw logits into a probability distribution by exponentiating and normalizing, used in classification and attention.\n\n**Signature:** `my_softmax(x, dim=-1) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor of any shape\n- `dim` — dimension along which to compute softmax\n\n**Returns:** probability tensor (sums to 1 along dim), same shape as input\n\n**Constraints:**\n- Subtract max for numerical stability before exp\n- Must handle large values without NaN/Inf",
    "description_zh": "实现 softmax 函数。\n\nSoftmax 通过指数化和归一化将原始 logits 转换为概率分布，用于分类和注意力机制。\n\n**签名:** `my_softmax(x, dim=-1) -> Tensor`\n\n**参数:**\n- `x` — 任意形状的输入张量\n- `dim` — 计算 softmax 的维度\n\n**返回:** 概率张量（沿 dim 求和为 1），形状与输入相同\n\n**约束:**\n- 在 exp 之前减去最大值以保证数值稳定\n- 必须处理大值而不产生 NaN/Inf",
    "function_name": "my_softmax",
    "hint": "1. `x_max = x.max(dim=dim, keepdim=True).values`\n2. `e_x = exp(x - x_max)`\n3. `return e_x / e_x.sum(dim=dim, keepdim=True)`",
    "hint_zh": "1. `x_max = x.max(dim=dim, keepdim=True).values`\n2. `e_x = exp(x - x_max)`\n3. `return e_x / e_x.sum(dim=dim, keepdim=True)`",
    "tests": [
        {
            "name": "Basic 1-D",
            "code": """
import torch
x = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
""",
        },
        {
            "name": "2-D along dim=-1",
            "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert out.shape == expected.shape, f'Shape mismatch'
assert torch.allclose(out, expected, atol=1e-5), 'Values differ'
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5), 'Rows must sum to 1'
""",
        },
        {
            "name": "Numerical stability",
            "code": """
import torch
x = torch.tensor([1000., 1001., 1002.])
out = {fn}(x, dim=-1)
assert not torch.isnan(out).any(), 'NaN in output — not numerically stable'
assert not torch.isinf(out).any(), 'Inf in output — not numerically stable'
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), 'Values differ on large input'
""",
        },
        {
            "name": "dim=0 softmax",
            "code": """
import torch
torch.manual_seed(3)
x = torch.randn(4, 3)
out = {fn}(x, dim=0)
exp_x = (x - x.max(dim=0, keepdim=True).values).exp()
expected = exp_x / exp_x.sum(dim=0, keepdim=True)
assert torch.allclose(out, expected, atol=1e-5), f'dim=0 softmax failed'
assert torch.allclose(out.sum(dim=0), torch.ones(3), atol=1e-5), 'columns should sum to 1'
""",
        },
    ],
    "solution": '''def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)''',
    "demo": """x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", my_softmax(x, dim=-1))
print("Sum:   ", my_softmax(x, dim=-1).sum())
print("Ref:   ", torch.softmax(x, dim=-1))""",

}