"""Flow Matching Loss task."""

TASK = {
    "title": "Flow Matching Loss",
    "title_zh": "流匹配损失",
    "difficulty": "Easy",
    "description_en": "Implement the flow matching training loss, the generative modeling paradigm behind Stable Diffusion 3, Flux, and Sora.\n\nFlow matching trains a neural network to predict the velocity field that transports noise to data along straight paths. The target velocity at any interpolated point is simply the direction from noise to data.\n\n**Signature:** `flow_matching_loss(model_output, x0, x1, t) -> Tensor`\n\n**Parameters:**\n- `model_output` — predicted velocity v_θ(x_t, t), shape (B, D)\n- `x0` — noise samples, shape (B, D)\n- `x1` — data samples, shape (B, D)\n- `t` — timesteps in [0, 1], shape (B,)\n\n**Returns:** scalar MSE loss\n\n**Formula:**\n- Target velocity: `u_t = x1 - x0` (straight-line direction from noise to data)\n- Loss: `mean(||model_output - u_t||²)`\n\nNote: the interpolated point `x_t = t * x1 + (1 - t) * x0` is computed externally before calling the model; `t` is passed here only for interface completeness.",
    "description_zh": "实现流匹配训练损失——Stable Diffusion 3、Flux 和 Sora 背后的生成建模范式。\n\n流匹配训练神经网络预测速度场，将噪声沿直线路径传输到数据。在任意插值点处，目标速度就是从噪声指向数据的方向。\n\n**签名:** `flow_matching_loss(model_output, x0, x1, t) -> Tensor`\n\n**参数:**\n- `model_output` — 预测速度 v_θ(x_t, t)，形状 (B, D)\n- `x0` — 噪声样本，形状 (B, D)\n- `x1` — 数据样本，形状 (B, D)\n- `t` — [0, 1] 内的时间步，形状 (B,)\n\n**返回:** 标量 MSE 损失\n\n**公式:**\n- 目标速度：`u_t = x1 - x0`（从噪声到数据的直线方向）\n- 损失：`mean(||model_output - u_t||²)`\n\n注意：插值点 `x_t = t * x1 + (1 - t) * x0` 在调用模型前已在外部计算；`t` 在此仅为接口完整性而传入。",
    "function_name": "flow_matching_loss",
    "hint": "target = x1 - x0  (straight-line velocity from noise to data)\nloss = ((model_output - target) ** 2).mean()\n`t` is not used in the loss itself.",
    "hint_zh": "target = x1 - x0（从噪声到数据的直线速度）\nloss = ((model_output - target) ** 2).mean()\n损失计算本身不需要 `t`。",
    "tests": [
        {
            "name": "Returns scalar",
            "code": """
import torch
torch.manual_seed(0)
B, D = 4, 32
model_output = torch.randn(B, D)
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
loss = {fn}(model_output, x0, x1, t)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'
"""
        },
        {
            "name": "Perfect prediction gives zero loss",
            "code": """
import torch
torch.manual_seed(1)
B, D = 8, 16
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
model_output = x1 - x0   # perfect velocity prediction
loss = {fn}(model_output, x0, x1, t)
assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f'Perfect prediction should give 0 loss, got {loss.item()}'
"""
        },
        {
            "name": "Loss is non-negative",
            "code": """
import torch
torch.manual_seed(2)
for _ in range(10):
    B, D = 4, 8
    model_output = torch.randn(B, D)
    x0 = torch.randn(B, D)
    x1 = torch.randn(B, D)
    t = torch.rand(B)
    loss = {fn}(model_output, x0, x1, t)
    assert loss.item() >= 0, f'Loss must be non-negative, got {loss.item()}'
"""
        },
        {
            "name": "Gradient flows through model_output",
            "code": """
import torch
B, D = 4, 16
model_output = torch.randn(B, D, requires_grad=True)
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
loss = {fn}(model_output, x0, x1, t)
loss.backward()
assert model_output.grad is not None, 'No gradient for model_output'
"""
        },
        {
            "name": "Exact numerical value",
            "code": """
import torch
torch.manual_seed(0)
B, D = 4, 8
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
model_output = torch.randn(B, D)
target = x1 - x0
expected = ((model_output - target) ** 2).mean()
out = {fn}(model_output, x0, x1, t)
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected.item():.6f}, got {out.item():.6f}'
""",
        },
        {
            "name": "Loss scales with prediction error",
            "code": """
import torch
torch.manual_seed(3)
B, D = 4, 8
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t = torch.rand(B)
target = x1 - x0
small_err = target + 0.01 * torch.randn(B, D)
large_err = target + 1.0 * torch.randn(B, D)
loss_small = {fn}(small_err, x0, x1, t)
loss_large = {fn}(large_err, x0, x1, t)
assert loss_small.item() < loss_large.item(), 'Larger prediction error should give larger loss'
"""
        }
    ],
    "solution": '''def flow_matching_loss(model_output, x0, x1, t):
    target_velocity = x1 - x0
    diff = model_output - target_velocity
    return (diff * diff).mean()''',
    "demo": """torch.manual_seed(0)

B, D = 16, 8
x0 = torch.randn(B, D)
x1 = torch.randn(B, D)
t  = torch.rand(B)

perfect_output = x1 - x0
loss_perfect = flow_matching_loss(perfect_output, x0, x1, t)
print(f"Perfect prediction => loss = {loss_perfect.item():.6f}  (expected 0.0)")

random_output = torch.randn(B, D)
loss_random = flow_matching_loss(random_output, x0, x1, t)
print(f"Random prediction  => loss = {loss_random.item():.4f}   (expected > 0)")

noisy_output = perfect_output + 0.1 * torch.randn(B, D)
loss_noisy = flow_matching_loss(noisy_output, x0, x1, t)
print(f"Noisy prediction   => loss = {loss_noisy.item():.4f}   (expected small but > 0)")""",

}
