"""GELU Activation task."""

TASK = {
    "title": "GELU Activation",
    "title_zh": "GELU 激活函数",
    "difficulty": "Easy",
    "description_en": "Implement the GELU activation function.\n\nGELU (Gaussian Error Linear Unit) smoothly gates inputs based on their value, used in transformers like BERT and GPT.\n\n**Signature:** `my_gelu(x) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor of any shape\n\n**Returns:** element-wise GELU activation, same shape as input\n\n**Constraints:**\n- Exact formula: `x * 0.5 * (1 + erf(x / sqrt(2)))`\n- Must match `F.gelu` within 1e-4\n- `gelu(0) = 0`",
    "description_zh": "实现 GELU 激活函数。\n\nGELU（高斯误差线性单元）根据输入值平滑地进行门控，广泛用于 BERT 和 GPT 等 Transformer。\n\n**签名:** `my_gelu(x) -> Tensor`\n\n**参数:**\n- `x` — 任意形状的输入张量\n\n**返回:** 逐元素 GELU 激活，形状与输入相同\n\n**约束:**\n- 精确公式：`x * 0.5 * (1 + erf(x / sqrt(2)))`\n- 必须与 `F.gelu` 误差在 1e-4 以内\n- `gelu(0) = 0`",
    "function_name": "my_gelu",
    "hint": "Exact: `0.5 * x * (1 + torch.erf(x / sqrt(2)))`\nApprox: `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`",
    "hint_zh": "精确版：`0.5 * x * (1 + torch.erf(x / sqrt(2)))`\n近似版：`0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`",
    "tests": [
        {
            "name": "Matches F.gelu",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(4, 8)\nout = {fn}(x)\nref = torch.nn.functional.gelu(x)\nassert torch.allclose(out, ref, atol=1e-4), 'Does not match F.gelu'\n"
        },
        {
            "name": "gelu(0) = 0",
            "code": "\nimport torch\nout = {fn}(torch.tensor([0.0]))\nassert torch.allclose(out, torch.tensor([0.0]), atol=1e-7), f'gelu(0) = {out.item()}'\n"
        },
        {
            "name": "Shape preservation",
            "code": "\nimport torch\nx = torch.randn(2, 3, 4)\nassert {fn}(x).shape == x.shape, 'Shape mismatch'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nx = torch.randn(4, 8, requires_grad=True)\n{fn}(x).sum().backward()\nassert x.grad is not None and x.grad.shape == x.shape, 'Gradient issue'\n"
        }
    ],
    "solution": '''def my_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))''',
    "demo": """x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', my_gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))""",

}