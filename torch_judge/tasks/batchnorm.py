"""BatchNorm implementation task."""

TASK = {
    "title": "Implement BatchNorm",
    "title_zh": "实现 BatchNorm",
    "difficulty": "Medium",
    "description_en": "Implement Batch Normalization with train/eval modes.\n\nBatchNorm normalizes activations per feature using batch statistics during training and running statistics during inference, stabilizing deep network training.\n\n**Signature:** `my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor (N, D)\n- `gamma`, `beta` — learnable affine parameters (D,)\n- `running_mean`, `running_var` — running statistics (D,), updated in-place during training\n\n**Returns:** normalized and affine-transformed tensor, same shape as x\n\n**Constraints:**\n- Training: use batch stats, update running stats with momentum\n- Inference: use running stats only\n- Use `unbiased=False` for batch variance",
    "description_zh": "实现带训练/推理模式的批归一化。\n\nBatchNorm 在训练时使用批统计量、推理时使用运行统计量对每个特征进行归一化，从而稳定深度网络训练。\n\n**签名:** `my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True) -> Tensor`\n\n**参数:**\n- `x` — 输入张量 (N, D)\n- `gamma`, `beta` — 可学习的仿射参数 (D,)\n- `running_mean`, `running_var` — 运行统计量 (D,)，训练时原地更新\n\n**返回:** 归一化并仿射变换后的张量，形状与 x 相同\n\n**约束:**\n- 训练模式：使用批统计量，用 momentum 更新运行统计量\n- 推理模式：仅使用运行统计量\n- 批方差使用 `unbiased=False`",
    "function_name": "my_batch_norm",
    "hint": "Train: `mean/var = x.mean/var(dim=0)`, update running stats with `momentum`\nEval: use `running_mean/running_var` directly\nBoth: `gamma * (x - mean) / sqrt(var + eps) + beta`",
    "hint_zh": "训练：`mean/var = x.mean/var(dim=0)`，用 `momentum` 更新运行统计量\n推理：直接使用 `running_mean/running_var`\n两者：`gamma * (x - mean) / sqrt(var + eps) + beta`",
    "tests": [
        {
            "name": "Training mode — zero mean per feature",
            "code": """
import torch
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)
running_mean = torch.zeros(4)
running_var = torch.ones(4)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
col_means = out.mean(dim=0)
assert torch.allclose(col_means, torch.zeros(4), atol=1e-5), f'Column means not zero: {col_means}'
""",
        },
        {
            "name": "Training mode — numerical correctness and running stats update",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(16, 8)
gamma = torch.randn(8)
beta = torch.randn(8)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
momentum = 0.1
out = {fn}(x, gamma, beta, running_mean, running_var, momentum=momentum, training=True)

# Reference using batch stats
mean = x.mean(dim=0)
var = x.var(dim=0, unbiased=False)
ref = gamma * (x - mean) / torch.sqrt(var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch'

# Running stats should have moved toward batch stats
expected_mean = (1 - momentum) * torch.zeros_like(mean) + momentum * mean
expected_var = (1 - momentum) * torch.ones_like(var) + momentum * var
assert torch.allclose(running_mean, expected_mean, atol=1e-6), 'running_mean not updated correctly'
assert torch.allclose(running_var, expected_var, atol=1e-6), 'running_var not updated correctly'
""",
        },
        {
            "name": "Inference mode — uses running statistics",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
gamma = torch.randn(8)
beta = torch.randn(8)

# Pretend these came from previous training
running_mean = torch.randn(8)
running_var = torch.rand(8) + 0.5  # positive

out = {fn}(x, gamma, beta, running_mean.clone(), running_var.clone(), training=False)
ref = gamma * (x - running_mean) / torch.sqrt(running_var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Inference should use running stats'
""",
        },
        {
            "name": "Gradient flow w.r.t inputs and affine params",
            "code": """
import torch
x = torch.randn(4, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
assert beta.grad is not None, 'beta.grad is None'
""",
        },
    ],
    "solution": '''import torch

def my_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    """BatchNorm with train/eval behavior and running stats.

    - Training: use batch stats, update running_mean / running_var in-place.
    - Inference: use running_mean / running_var as-is.
    """
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        # Update running statistics in-place. Detach to avoid tracking gradients.
        running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
        running_var.mul_(1 - momentum).add_(momentum * batch_var.detach())

        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta''',
    "demo": """x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)

running_mean = torch.zeros(4)
running_var = torch.ones(4)

out_train = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
print("[Train] Column means:", out_train.mean(dim=0))
print("[Train] Column stds: ", out_train.std(dim=0))
print("Updated running_mean:", running_mean)
print("Updated running_var:", running_var)

out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
print("[Eval] Column means (using running stats):", out_eval.mean(dim=0))""",

}