"""DDIM Sampling Step task."""

TASK = {
    "title": "DDIM Sampling Step",
    "title_zh": "DDIM 采样步骤",
    "difficulty": "Medium",
    "description_en": "Implement one step of DDIM (Denoising Diffusion Implicit Models) deterministic sampling.\n\nGiven the current noisy sample and the model's noise prediction, compute the previous (less noisy) sample without any stochastic noise injection.\n\n**Signature:** `ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev) -> Tensor`\n\n**Parameters:**\n- `x_t` — current noisy sample at timestep t, shape (B, ...)\n- `noise_pred` — model's predicted noise ε_θ(x_t, t), same shape as x_t\n- `alpha_bar_t` — scalar, cumulative noise schedule ᾱ_t at current step\n- `alpha_bar_prev` — scalar, cumulative noise schedule ᾱ_{t-1} at previous step\n\n**Returns:** x_{t-1} with same shape as x_t\n\n**Formula:**\n1. Predict x0: `x0_pred = (x_t - sqrt(1 - ᾱ_t) * noise_pred) / sqrt(ᾱ_t)`\n2. Direction toward x_t: `noise_direction = sqrt(1 - ᾱ_{t-1}) * noise_pred`\n3. `x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + noise_direction`",
    "description_zh": "实现 DDIM（去噪扩散隐式模型）的单步确定性采样。\n\n给定当前含噪样本和模型的噪声预测，在不注入随机噪声的情况下计算上一步（噪声更少的）样本。\n\n**签名:** `ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev) -> Tensor`\n\n**参数:**\n- `x_t` — 时间步 t 的含噪样本，形状 (B, ...)\n- `noise_pred` — 模型预测的噪声 ε_θ(x_t, t)，形状与 x_t 相同\n- `alpha_bar_t` — 标量，当前步的累积噪声调度 ᾱ_t\n- `alpha_bar_prev` — 标量，上一步的累积噪声调度 ᾱ_{t-1}\n\n**返回:** x_{t-1}，形状与 x_t 相同\n\n**公式:**\n1. 预测 x0：`x0_pred = (x_t - sqrt(1 - ᾱ_t) * noise_pred) / sqrt(ᾱ_t)`\n2. 指向 x_t 的方向：`noise_direction = sqrt(1 - ᾱ_{t-1}) * noise_pred`\n3. `x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + noise_direction`",
    "function_name": "ddim_step",
    "hint": "1. x0_pred = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t\n2. noise_dir = √(1-ᾱ_{t-1})·ε\n3. x_{t-1} = √ᾱ_{t-1}·x0_pred + noise_dir\n   Use ** 0.5 for square roots.",
    "hint_zh": "1. x0_pred = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t\n2. noise_dir = √(1-ᾱ_{t-1})·ε\n3. x_{t-1} = √ᾱ_{t-1}·x0_pred + noise_dir\n   用 ** 0.5 计算平方根。",
    "tests": [
        {
            "name": "Output shape matches x_t",
            "code": """
import torch
torch.manual_seed(0)
x_t = torch.randn(4, 3, 8, 8)
noise_pred = torch.randn(4, 3, 8, 8)
out = {fn}(x_t, noise_pred, alpha_bar_t=0.5, alpha_bar_prev=0.7)
assert out.shape == x_t.shape, f'Expected {x_t.shape}, got {out.shape}'
"""
        },
        {
            "name": "alpha_bar_prev=1 recovers x0_pred",
            "code": """
import torch
torch.manual_seed(1)
x_t = torch.randn(2, 16)
noise_pred = torch.randn(2, 16)
alpha_bar_t = 0.4
# When alpha_bar_prev=1: noise_direction=0, x_prev = 1*x0_pred = x0_pred
x_prev = {fn}(x_t, noise_pred, alpha_bar_t=alpha_bar_t, alpha_bar_prev=1.0)
x0_pred = (x_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
assert torch.allclose(x_prev, x0_pred, atol=1e-5), 'With alpha_bar_prev=1, output should equal x0_pred'
"""
        },
        {
            "name": "alpha_bar_t == alpha_bar_prev gives x_t back",
            "code": """
import torch
torch.manual_seed(2)
x_t = torch.randn(3, 32)
noise_pred = torch.randn(3, 32)
alpha_bar = 0.6
x_prev = {fn}(x_t, noise_pred, alpha_bar_t=alpha_bar, alpha_bar_prev=alpha_bar)
assert torch.allclose(x_prev, x_t, atol=1e-5), 'When alpha_bar_t == alpha_bar_prev, output should equal x_t'
"""
        },
        {
            "name": "Gradient flow through x_t and noise_pred",
            "code": """
import torch
x_t = torch.randn(2, 8, requires_grad=True)
noise_pred = torch.randn(2, 8, requires_grad=True)
out = {fn}(x_t, noise_pred, alpha_bar_t=0.5, alpha_bar_prev=0.3)
out.sum().backward()
assert x_t.grad is not None, 'No gradient for x_t'
assert noise_pred.grad is not None, 'No gradient for noise_pred'
"""
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch
torch.manual_seed(3)
x_t = torch.randn(1, 4)
noise_pred = torch.randn(1, 4)
abt, abp = 0.64, 0.81
x0_pred = (x_t - (1 - abt) ** 0.5 * noise_pred) / (abt ** 0.5)
noise_dir = (1 - abp) ** 0.5 * noise_pred
expected = abp ** 0.5 * x0_pred + noise_dir
result = {fn}(x_t, noise_pred, alpha_bar_t=abt, alpha_bar_prev=abp)
assert torch.allclose(result, expected, atol=1e-5), f'Numerical mismatch: {result} vs {expected}'
"""
        }
    ],
    "solution": '''def ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev):
    # Predict x0
    x0_pred = (x_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
    # Direction toward x_t
    noise_direction = (1 - alpha_bar_prev) ** 0.5 * noise_pred
    # Previous sample
    x_prev = (alpha_bar_prev ** 0.5) * x0_pred + noise_direction
    return x_prev''',
    "demo": """torch.manual_seed(42)

T = 5
alpha_bars = torch.linspace(0.99, 0.01, T + 1)  # index 0..T

x_clean = torch.tensor([1.0, -1.0, 0.5])  # target signal
noise   = torch.randn_like(x_clean)

ab_T = alpha_bars[T]
x_t  = ab_T ** 0.5 * x_clean + (1 - ab_T) ** 0.5 * noise

print(f"{'Step':>4}  {'alpha_bar_t':>12}  {'x_t (first elem)':>18}")
print("-" * 42)
for step in range(T, 0, -1):
    ab_t    = alpha_bars[step]
    ab_prev = alpha_bars[step - 1]
    noise_pred = (x_t - ab_t ** 0.5 * x_clean) / (1 - ab_t) ** 0.5
    x_t = ddim_step(x_t, noise_pred, ab_t, ab_prev)
    print(f"{step:>4}  {ab_prev.item():>12.4f}  {x_t[0].item():>18.4f}")

print(f"\nFinal x vs clean: {x_t.tolist()}  vs  {x_clean.tolist()}")""",

}
