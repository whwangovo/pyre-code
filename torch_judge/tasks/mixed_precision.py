"""Mixed Precision Training Step task."""

TASK = {
    "title": "Mixed Precision Training Step",
    "title_zh": "混合精度训练步骤",
    "difficulty": "Medium",
    "description_en": "Implement one mixed precision training step.\n\nMixed precision training uses fp16 for the forward/backward pass (faster, less memory) while keeping fp32 master weights for numerical stability. Loss scaling prevents fp16 gradient underflow.\n\n**Signature:** `mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0) -> float`\n\n**Parameters:**\n- `model` — nn.Module with fp32 parameters\n- `optimizer` — optimizer holding fp32 params\n- `loss_fn` — callable `(output, target) -> scalar loss`\n- `x, y` — input and target tensors\n- `loss_scale` — scalar to multiply loss before backward\n\n**Returns:** unscaled loss value as float\n\n**Steps:**\n1. Cast model to fp16, run forward pass\n2. Compute loss, scale it by `loss_scale`\n3. Run backward on scaled loss\n4. Unscale gradients (divide by `loss_scale`)\n5. Update optimizer (fp32 master weights)\n6. Restore model to fp32\n7. Return original (unscaled) loss",
    "description_zh": "实现一步混合精度训练。\n\n混合精度训练用 fp16 做前向/反向传播（更快、更省内存），同时保留 fp32 主权重以保证数值稳定性。Loss scaling 防止 fp16 梯度下溢。\n\n**签名:** `mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0) -> float`\n\n**参数:**\n- `model` — 持有 fp32 参数的 nn.Module\n- `optimizer` — 持有 fp32 参数的优化器\n- `loss_fn` — 可调用对象 `(output, target) -> 标量损失`\n- `x, y` — 输入和目标张量\n- `loss_scale` — 反向传播前乘以 loss 的缩放系数\n\n**返回:** 未缩放的 loss 值（float）\n\n**步骤:**\n1. 将模型转为 fp16，运行前向传播\n2. 计算 loss，乘以 `loss_scale`\n3. 对缩放后的 loss 做反向传播\n4. 反缩放梯度（除以 `loss_scale`）\n5. 更新优化器（fp32 主权重）\n6. 将模型恢复为 fp32\n7. 返回原始（未缩放）loss",
    "function_name": "mixed_precision_step",
    "hint": "1. model.half() → forward with fp16 input\n2. compute loss → scale by loss_scale → backward\n3. divide all param.grad by loss_scale (unscale)\n4. model.float() → optimizer.step()\n5. return unscaled loss as float",
    "hint_zh": "1. model.half() → fp16 输入前向传播\n2. 计算 loss → 乘以 loss_scale → 反向传播\n3. 所有 param.grad 除以 loss_scale（反缩放）\n4. model.float() → optimizer.step()\n5. 返回未缩放的 loss（float）",
    "tests": [
        {
            "name": "Returns float loss",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(8, 4)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(4, 8)
y = torch.randn(4, 4)
loss_val = {fn}(model, opt, nn.functional.mse_loss, x, y)
assert isinstance(loss_val, float), f'Expected float, got {type(loss_val)}'
assert loss_val > 0, 'Loss should be positive'
""",
        },
        {
            "name": "Parameters are updated",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(1)
model = nn.Linear(8, 4)
w_before = model.weight.data.clone()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(4, 8)
y = torch.randn(4, 4)
{fn}(model, opt, nn.functional.mse_loss, x, y)
assert not torch.allclose(model.weight.data, w_before), 'Weights should be updated after step'
""",
        },
        {
            "name": "Model restored to fp32 after step",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(2)
model = nn.Linear(8, 4)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(4, 8)
y = torch.randn(4, 4)
{fn}(model, opt, nn.functional.mse_loss, x, y)
for p in model.parameters():
    assert p.dtype == torch.float32, f'Parameter should be fp32 after step, got {p.dtype}'
""",
        },
        {
            "name": "fp16 used in forward pass",
            "code": """
import torch
import torch.nn as nn

class DtypeCapture(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.captured_dtype = None
    def forward(self, x):
        self.captured_dtype = x.dtype
        return self.linear(x)

torch.manual_seed(0)
model = nn.Sequential(DtypeCapture(nn.Linear(8, 8)), nn.Linear(8, 4))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(2, 8)
y = torch.randn(2, 4)
loss_fn = nn.MSELoss()
{fn}(model, optimizer, loss_fn, x, y)
assert model[0].captured_dtype == torch.float16, f'Expected fp16 in forward, got {model[0].captured_dtype}'
""",
        },
        {
            "name": "Loss scale affects gradients",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(3)
model1 = nn.Linear(4, 2)
model2 = nn.Linear(4, 2)
model2.load_state_dict(model1.state_dict())
opt1 = torch.optim.SGD(model1.parameters(), lr=0.0)  # lr=0 to keep grads
opt2 = torch.optim.SGD(model2.parameters(), lr=0.0)
x = torch.randn(2, 4)
y = torch.randn(2, 2)
{fn}(model1, opt1, nn.functional.mse_loss, x, y, loss_scale=1.0)
{fn}(model2, opt2, nn.functional.mse_loss, x, y, loss_scale=1024.0)
# After unscaling, gradients should be the same regardless of loss_scale
for p1, p2 in zip(model1.parameters(), model2.parameters()):
    if p1.grad is not None and p2.grad is not None:
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), 'Unscaled grads should match regardless of loss_scale'
""",
        },
    ],
    "solution": '''def mixed_precision_step(model, optimizer, loss_fn, x, y, loss_scale=1024.0):
    # 1. Cast to fp16 for forward pass
    model.half()
    x_fp16 = x.half()
    with torch.no_grad():
        pass  # weights already fp16
    output = model(x_fp16)
    loss = loss_fn(output.float(), y)
    loss_val = loss.item()

    # 2. Scale loss and backward
    optimizer.zero_grad()
    (loss * loss_scale).backward()

    # 3. Unscale gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = p.grad.data.float() / loss_scale

    # 4. Update (fp32 master weights via optimizer)
    model.float()
    optimizer.step()

    return loss_val''',
    "demo": """torch.manual_seed(42)
model = nn.Linear(8, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(4, 8)
y = torch.randn(4, 4)

weights_before = model.weight.data.clone()

loss_val = mixed_precision_step(model, optimizer, loss_fn, x, y)

print("Loss:", loss_val)
print("Weights updated:", not torch.allclose(model.weight.data, weights_before))
print("Model dtype after step:", model.weight.dtype)""",

}
