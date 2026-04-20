"""Diffusion Noise Schedules task."""

TASK = {
    "title": "Diffusion Noise Schedules",
    "title_zh": "扩散噪声调度",
    "difficulty": "Easy",
    "description_en": "Implement three noise schedules used in diffusion models.\n\nEach schedule returns `alpha_bar` (ᾱ), the cumulative product of (1 - β) across timesteps. ᾱ_t controls the signal-to-noise ratio: close to 1 means mostly signal, close to 0 means mostly noise.\n\n**Signature:** `noise_schedule(num_timesteps, schedule_type='cosine') -> Tensor`\n\n**Parameters:**\n- `num_timesteps` — number of diffusion timesteps T\n- `schedule_type` — one of `'linear'`, `'cosine'`, or `'sigmoid'`\n\n**Returns:** `alpha_bar` tensor of shape `(num_timesteps,)`, values in (0, 1], monotonically decreasing\n\n**Formulas:**\n\n`linear`: β_t = β_start + (β_end − β_start) × t/(T−1), then ᾱ = cumprod(1 − β)\n- Use β_start=1e-4, β_end=0.02\n\n`cosine`: ᾱ_t = cos²((t/T + s)/(1+s) × π/2) / cos²(s/(1+s) × π/2)\n- Use s=0.008, clip to [0.0001, 0.9999]\n\n`sigmoid`: β_t = sigmoid(t/T × 12 − 6), scaled to [β_start, β_end], then ᾱ = cumprod(1 − β)\n- Use β_start=1e-4, β_end=0.02",
    "description_zh": "实现扩散模型中使用的三种噪声调度。\n\n每种调度返回 `alpha_bar`（ᾱ），即各时间步 (1 - β) 的累积乘积。ᾱ_t 控制信噪比：接近 1 表示主要是信号，接近 0 表示主要是噪声。\n\n**签名:** `noise_schedule(num_timesteps, schedule_type='cosine') -> Tensor`\n\n**参数:**\n- `num_timesteps` — 扩散时间步数 T\n- `schedule_type` — `'linear'`、`'cosine'` 或 `'sigmoid'` 之一\n\n**返回:** 形状为 `(num_timesteps,)` 的 `alpha_bar` 张量，值域 (0, 1]，单调递减\n\n**公式:**\n\n`linear`：β_t = β_start + (β_end − β_start) × t/(T−1)，然后 ᾱ = cumprod(1 − β)\n- 使用 β_start=1e-4，β_end=0.02\n\n`cosine`：ᾱ_t = cos²((t/T + s)/(1+s) × π/2) / cos²(s/(1+s) × π/2)\n- 使用 s=0.008，截断到 [0.0001, 0.9999]\n\n`sigmoid`：β_t = sigmoid(t/T × 12 − 6)，缩放到 [β_start, β_end]，然后 ᾱ = cumprod(1 − β)\n- 使用 β_start=1e-4，β_end=0.02",
    "function_name": "noise_schedule",
    "hint": "linear/sigmoid: compute betas → alpha_bar = torch.cumprod(1-betas, dim=0)\ncosine: f(t) = cos(...)² → alpha_bar = f(t)/f(0), clamp to [0.0001, 0.9999]\nsigmoid scaling: shift+scale so min(sig)→β_start, max(sig)→β_end",
    "hint_zh": "linear/sigmoid：计算 betas → alpha_bar = torch.cumprod(1-betas, dim=0)\ncosine：f(t) = cos(...)² → alpha_bar = f(t)/f(0)，clamp 到 [0.0001, 0.9999]\nsigmoid 缩放：平移+缩放使 min(sig)→β_start，max(sig)→β_end",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(100, stype)
    assert out.shape == (100,), f'{stype}: expected shape (100,), got {out.shape}'
"""
        },
        {
            "name": "All values in (0, 1]",
            "code": """
import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(200, stype)
    assert (out > 0).all(), f'{stype}: alpha_bar must be > 0, min={out.min().item()}'
    assert (out <= 1.0).all(), f'{stype}: alpha_bar must be <= 1, max={out.max().item()}'
"""
        },
        {
            "name": "Monotonically decreasing",
            "code": """
import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(100, stype)
    diffs = out[1:] - out[:-1]
    assert (diffs <= 1e-6).all(), f'{stype}: alpha_bar must be monotonically decreasing'
"""
        },
        {
            "name": "alpha_bar[0] close to 1 (low noise at start)",
            "code": """
import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(1000, stype)
    assert out[0].item() > 0.9, f'{stype}: alpha_bar[0] should be close to 1, got {out[0].item()}'
"""
        },
        {
            "name": "alpha_bar[-1] close to 0 (high noise at end)",
            "code": """
import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(1000, stype)
    assert out[-1].item() < 0.05, f'{stype}: alpha_bar[-1] should be close to 0, got {out[-1].item()}'
"""
        },
        {
            "name": "All three schedule types work without error",
            "code": """
import torch
for stype in ['linear', 'cosine', 'sigmoid']:
    out = {fn}(50, stype)
    assert isinstance(out, torch.Tensor), f'{stype}: must return a torch.Tensor'
    assert out.dtype == torch.float32, f'{stype}: expected float32, got {out.dtype}'
try:
    {fn}(10, 'unknown')
    assert False, 'Should raise ValueError for unknown schedule_type'
except (ValueError, Exception):
    pass
"""
        },
        {
            "name": "Linear schedule exact values",
            "code": """
import torch
T = 5
out = {fn}(T, schedule_type='linear')
beta_start, beta_end = 1e-4, 0.02
t = torch.arange(T, dtype=torch.float32)
betas = beta_start + (beta_end - beta_start) * t / (T - 1)
alpha_bars = torch.cumprod(1.0 - betas, dim=0)
assert torch.allclose(out, alpha_bars, atol=1e-5), f'Linear schedule mismatch.\\nExpected: {alpha_bars}\\nGot: {out}'
"""
        }
    ],
    "solution": '''import math as _math

def noise_schedule(num_timesteps, schedule_type=\'cosine\'):
    T = num_timesteps
    t = torch.arange(T, dtype=torch.float32)
    if schedule_type == \'linear\':
        beta_start, beta_end = 1e-4, 0.02
        betas = beta_start + (beta_end - beta_start) * t / (T - 1)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
    elif schedule_type == \'cosine\':
        s = 0.008
        f = torch.cos(((t / T + s) / (1 + s)) * (_math.pi / 2)) ** 2
        f0 = _math.cos((s / (1 + s)) * (_math.pi / 2)) ** 2
        alpha_bar = (f / f0).clamp(0.0001, 0.9999)
    elif schedule_type == \'sigmoid\':
        beta_start, beta_end = 1e-4, 0.02
        x = t / T * 12 - 6
        sig = 1 / (1 + torch.exp(-x))
        betas = beta_start + (beta_end - beta_start) * (sig - sig.min()) / (sig.max() - sig.min())
        alpha_bar = torch.cumprod(1 - betas, dim=0)
    else:
        raise ValueError(f\'Unknown schedule_type: {schedule_type}\')
    return alpha_bar''',
    "demo": """T = 1000
schedules = ['linear', 'cosine', 'sigmoid']
checkpoints = [0, T // 4, T // 2, 3 * T // 4, T - 1]
labels = ['t=0', 'T/4', 'T/2', '3T/4', 'T-1']

col_w = 10
header = f"{'t':>6}" + "".join(f"{s:>{col_w}}" for s in schedules)
print(header)
print("-" * len(header))

results = {s: noise_schedule(T, s) for s in schedules}
for label, idx in zip(labels, checkpoints):
    row = f"{label:>6}" + "".join(f"{results[s][idx].item():>{col_w}.4f}" for s in schedules)
    print(row)""",

}
