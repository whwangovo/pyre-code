"""Implement Dropout task."""

TASK = {
    "title": "Implement Dropout",
    "title_zh": "实现 Dropout",
    "difficulty": "Easy",
    "description_en": "Implement dropout as an nn.Module.\n\nDropout randomly zeroes elements during training and scales survivors by `1/(1-p)` to maintain expected values. During eval, it is an identity.\n\n**Signature:** `MyDropout(p=0.5)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor of any shape\n\n**Returns:** tensor with dropout applied (training) or unchanged (eval)\n\n**Constraints:**\n- Training: zero with probability p, scale by `1/(1-p)`\n- Eval: return input unchanged",
    "description_zh": "实现 Dropout（nn.Module）。\n\nDropout 在训练时以概率 p 随机将元素置零，并将存活元素缩放 `1/(1-p)` 以保持期望值不变。推理时为恒等映射。\n\n**签名:** `MyDropout(p=0.5)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 任意形状的输入张量\n\n**返回:** 应用 dropout 后的张量（训练）或原始输入（推理）\n\n**约束:**\n- 训练模式：以概率 p 置零，缩放 `1/(1-p)`\n- 推理模式：返回原始输入",
    "function_name": "MyDropout",
    "hint": "Train: `mask = (rand_like(x) > p).float()` → `x * mask / (1-p)`\nEval: return `x` unchanged",
    "hint_zh": "训练：`mask = (rand_like(x) > p).float()` → `x * mask / (1-p)`\n推理：直接返回 `x`",
    "tests": [
        {
            "name": "Eval mode is identity",
            "code": "\nimport torch, torch.nn as nn\nd = {fn}(p=0.5)\nassert isinstance(d, nn.Module), 'Must inherit from nn.Module'\nd.eval()\nx = torch.randn(4, 8)\nassert torch.equal(d(x), x), 'eval mode should return input unchanged'\n"
        },
        {
            "name": "Training: zeros and scaling",
            "code": "\nimport torch\ntorch.manual_seed(42)\nd = {fn}(p=0.5)\nd.train()\nx = torch.ones(1000)\nout = d(x)\nassert (out == 0).any(), 'No zeros found during training'\nnon_zero = out[out != 0]\nassert torch.allclose(non_zero, torch.full_like(non_zero, 2.0), atol=1e-5), 'Non-zeros should be scaled by 1/(1-p)=2.0'\n"
        },
        {
            "name": "Drop rate is approximately p",
            "code": "\nimport torch\ntorch.manual_seed(0)\nd = {fn}(p=0.3)\nd.train()\nout = d(torch.ones(10000))\nfrac = (out == 0).float().mean().item()\nassert 0.25 < frac < 0.35, f'Expected ~30%% zeros, got {frac*100:.1f}%%'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nd = {fn}(p=0.5)\nd.train()\nx = torch.randn(4, 8, requires_grad=True)\nd(x).sum().backward()\nassert x.grad is not None, 'x.grad is None'\n"
        }
    ],
    "solution": '''class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)''',
    "demo": """d = MyDropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))""",

}