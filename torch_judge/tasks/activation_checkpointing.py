"""Activation Checkpointing task."""

TASK = {
    "title": "Activation Checkpointing",
    "title_zh": "激活检查点",
    "difficulty": "Medium",
    "description_en": "Implement gradient checkpointing for a sequence of functions.\n\nActivation checkpointing trades compute for memory: instead of storing all intermediate activations during the forward pass, they are recomputed on-the-fly during backpropagation.\n\n**Signature:** `checkpoint_sequential(fns, x) -> Tensor`\n\n**Parameters:**\n- `fns` — list of callables, each mapping Tensor -> Tensor\n- `x` — input tensor\n\n**Returns:** output tensor after applying all functions sequentially\n\n**Constraints:**\n- Intermediate activations must NOT be stored during the forward pass\n- Gradients must flow back to `x`\n- Output must be numerically identical to naive sequential application\n- Use `torch.utils.checkpoint.checkpoint` (allowed — it is not `F.*` or `nn.functional.*`)",
    "description_zh": "为一系列函数实现梯度检查点。\n\n激活检查点以计算换内存：在前向传播期间不存储所有中间激活，而是在反向传播时按需重新计算。\n\n**签名:** `checkpoint_sequential(fns, x) -> Tensor`\n\n**参数:**\n- `fns` — 可调用对象列表，每个接受 Tensor 并返回 Tensor\n- `x` — 输入张量\n\n**返回:** 依次应用所有函数后的输出张量\n\n**约束:**\n- 前向传播期间不得存储中间激活\n- 梯度必须能流回 `x`\n- 输出必须与朴素顺序应用在数值上完全一致\n- 使用 `torch.utils.checkpoint.checkpoint`（允许——它不是 `F.*` 或 `nn.functional.*`）",
    "function_name": "checkpoint_sequential",
    "hint": "import torch.utils.checkpoint as cp\nLoop over fns: x = cp.checkpoint(fn, x, use_reentrant=False)\nActivations are recomputed on backward instead of stored.",
    "hint_zh": "import torch.utils.checkpoint as cp\n遍历 fns：x = cp.checkpoint(fn, x, use_reentrant=False)\n激活在反向传播时重新计算，而非存储。",
    "tests": [
        {
            "name": "Output matches naive sequential",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(42)
fns = [nn.Linear(16, 16) for _ in range(4)]
for fn in fns:
    fn.eval()
x = torch.randn(2, 16)
expected = x
for fn in fns:
    expected = fn(expected)
result = {fn}(fns, x)
assert result.shape == expected.shape, f'Shape mismatch: {result.shape}'
assert torch.allclose(result, expected, atol=1e-5), 'Output does not match naive sequential'
""",
        },
        {
            "name": "Gradient flows to input",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
fns = [nn.Linear(8, 8) for _ in range(3)]
x = torch.randn(1, 8, requires_grad=True)
out = {fn}(fns, x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None — gradient did not flow to input'
assert x.grad.shape == x.shape, f'x.grad shape mismatch: {x.grad.shape}'
""",
        },
        {
            "name": "Works with 4 linear layers",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(7)
fns = [nn.Linear(32, 32) for _ in range(4)]
x = torch.randn(4, 32, requires_grad=True)
out = {fn}(fns, x)
out.sum().backward()
assert out.shape == (4, 32), f'Output shape: {out.shape}'
assert x.grad is not None, 'No gradient on input'
""",
        },
        {
            "name": "Numerically identical to non-checkpointed version",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(3)
fns = [nn.Linear(16, 16) for _ in range(3)]
for fn in fns:
    fn.eval()
x = torch.randn(2, 16)
naive = x.clone()
for fn in fns:
    naive = fn(naive)
ckpt = {fn}(fns, x)
assert torch.allclose(naive, ckpt, atol=1e-5), f'Max diff: {(naive - ckpt).abs().max().item()}'
""",
        },
    ],
    "solution": """import torch.utils.checkpoint as cp

def checkpoint_sequential(fns, x):
    for fn in fns:
        x = cp.checkpoint(fn, x, use_reentrant=False)
    return x""",
    "demo": """torch.manual_seed(0)
layers = [torch.nn.Linear(16, 16) for _ in range(4)]
x = torch.randn(4, 16, requires_grad=True)

out_cp = checkpoint_sequential(layers, x)

x2 = x.detach().requires_grad_(True)
out_naive = x2
for layer in layers:
    out_naive = layer(out_naive)

print("Outputs match:", torch.allclose(out_cp, out_naive, atol=1e-5))

out_cp.sum().backward()
print("Gradient flows (x.grad is not None):", x.grad is not None)""",

}
