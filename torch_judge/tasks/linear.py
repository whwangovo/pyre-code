"""Simple Linear Layer task."""

TASK = {
    "title": "Simple Linear Layer",
    "title_zh": "简单线性层",
    "difficulty": "Medium",
    "description_en": "Implement a simple linear (fully connected) layer from scratch.\n\nA linear layer computes `y = x @ W^T + b` with learnable weight and bias tensors.\n\n**Signature:** `SimpleLinear(in_features, out_features)` (class)\n\n**Method:** `forward(x) -> Tensor`\n- `x` — input tensor (*, in_features)\n\n**Returns:** output tensor (*, out_features)\n\n**Constraints:**\n- Weight shape: (out_features, in_features) with Kaiming scaling\n- Bias shape: (out_features,) initialized to zeros\n- Both must have `requires_grad=True`",
    "description_zh": "从零实现简单线性（全连接）层。\n\n线性层使用可学习的权重和偏置张量计算 `y = x @ W^T + b`。\n\n**签名:** `SimpleLinear(in_features, out_features)`（类）\n\n**方法:** `forward(x) -> Tensor`\n- `x` — 输入张量 (*, in_features)\n\n**返回:** 输出张量 (*, out_features)\n\n**约束:**\n- 权重形状：(out_features, in_features)，使用 Kaiming 缩放\n- 偏置形状：(out_features,)，初始化为零\n- 两者都必须 `requires_grad=True`",
    "function_name": "SimpleLinear",
    "hint": "`weight` shape `(out, in)`, init `randn * 1/sqrt(in_features)`\n`bias` shape `(out,)`, init zeros\nForward: `x @ weight.T + bias`",
    "hint_zh": "`weight` 形状 `(out, in)`，初始化 `randn * 1/sqrt(in_features)`\n`bias` 形状 `(out,)`，初始化为零\n前向：`x @ weight.T + bias`",
    "tests": [
        {
            "name": "Weight & bias shape",
            "code": """
import torch
layer = {fn}(8, 4)
assert layer.weight.shape == (4, 8), f'Weight shape: {layer.weight.shape}'
assert layer.bias.shape == (4,), f'Bias shape: {layer.bias.shape}'
assert layer.weight.requires_grad, 'weight must require grad'
assert layer.bias.requires_grad, 'bias must require grad'
""",
        },
        {
            "name": "Forward pass",
            "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
assert y.shape == (2, 4), f'Output shape: {y.shape}'
expected = x @ layer.weight.T + layer.bias
assert torch.allclose(y, expected, atol=1e-5), 'Forward != x @ W^T + b'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
y.sum().backward()
assert layer.weight.grad is not None, 'weight.grad is None'
assert layer.bias.grad is not None, 'bias.grad is None'
""",
        },
    ],
    "solution": '''class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias''',
    "demo": """layer = SimpleLinear(8, 4)
print("W shape:", layer.weight.shape)
print("b shape:", layer.bias.shape)
x = torch.randn(2, 8)
print("Output shape:", layer.forward(x).shape)""",

}