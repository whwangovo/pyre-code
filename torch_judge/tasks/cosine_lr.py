"""Cosine LR Scheduler with Warmup task."""

TASK = {
    "title": "Cosine LR Scheduler with Warmup",
    "title_zh": "余弦学习率调度（含预热）",
    "difficulty": "Medium",
    "description_en": "Implement a cosine learning rate schedule with linear warmup.\n\nThis scheduler linearly ramps the LR during warmup, then decays it following a cosine curve. Widely used in transformer training.\n\n**Signature:** `cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0) -> float`\n\n**Parameters:**\n- `step` — current training step\n- `total_steps` — total number of steps\n- `warmup_steps` — number of warmup steps\n- `max_lr`, `min_lr` — peak and minimum learning rates\n\n**Returns:** learning rate as a float\n\n**Constraints:**\n- Warmup: linear from 0 to max_lr\n- Decay: `min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))`",
    "description_zh": "实现带线性预热的余弦学习率调度。\n\n该调度器在预热阶段线性增加学习率，之后按余弦曲线衰减，广泛用于 Transformer 训练。\n\n**签名:** `cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0) -> float`\n\n**参数:**\n- `step` — 当前训练步数\n- `total_steps` — 总步数\n- `warmup_steps` — 预热步数\n- `max_lr`, `min_lr` — 峰值和最小学习率\n\n**返回:** 浮点数学习率\n\n**约束:**\n- 预热阶段：从 0 线性增加到 max_lr\n- 衰减阶段：`min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))`",
    "function_name": "cosine_lr_schedule",
    "hint": "1. step < warmup_steps → lr = max_lr * step / warmup_steps\n2. step >= total_steps → lr = min_lr\n3. progress = (step - warmup_steps) / (total_steps - warmup_steps)\n   → min_lr + 0.5*(max_lr-min_lr)*(1 + cos(π·progress))",
    "hint_zh": "1. step < warmup_steps → lr = max_lr * step / warmup_steps\n2. step >= total_steps → lr = min_lr\n3. progress = (step - warmup_steps) / (total_steps - warmup_steps)\n   → min_lr + 0.5*(max_lr-min_lr)*(1 + cos(π·progress))",
    "tests": [
        {
            "name": "Start of warmup",
            "code": "\nlr = {fn}(step=0, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)\nassert abs(lr) < 1e-8, f'lr at step 0: {lr}'\n"
        },
        {
            "name": "End of warmup",
            "code": "\nlr = {fn}(step=10, total_steps=100, warmup_steps=10, max_lr=0.001)\nassert abs(lr - 0.001) < 1e-8, f'lr at warmup end: {lr}'\n"
        },
        {
            "name": "End of schedule",
            "code": "\nlr = {fn}(step=100, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0001)\nassert abs(lr - 0.0001) < 1e-6, f'lr at end: {lr}'\n"
        },
        {
            "name": "Warmup is monotonically increasing",
            "code": "\nlrs = [{fn}(step=i, total_steps=100, warmup_steps=10, max_lr=0.001) for i in range(11)]\nfor i in range(len(lrs) - 1):\n    assert lrs[i] <= lrs[i+1] + 1e-10, f'Not increasing at step {i}'\n"
        },
        {
            "name": "Cosine shape",
            "code": "\nimport math\nlr = {fn}(step=55, total_steps=100, warmup_steps=10, max_lr=0.001, min_lr=0.0)\nprogress = (55 - 10) / (100 - 10)\nexpected = 0.5 * 0.001 * (1 + math.cos(math.pi * progress))\nassert abs(lr - expected) < 1e-8, f'{lr} vs {expected}'\n"
        }
    ],
    "solution": '''def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))''',
    "demo": """lrs = [cosine_lr_schedule(i, 100, 10, 0.001) for i in range(101)]
print(f'Start: {lrs[0]:.6f}, Warmup end: {lrs[10]:.6f}, Mid: {lrs[55]:.6f}, End: {lrs[100]:.6f}')""",

}