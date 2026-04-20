"""Gradient Norm Clipping task."""

TASK = {
    "title": "Gradient Norm Clipping",
    "title_zh": "梯度范数裁剪",
    "difficulty": "Easy",
    "description_en": "Implement gradient norm clipping.\n\nGradient clipping rescales all parameter gradients when their combined L2 norm exceeds a threshold, preventing exploding gradients.\n\n**Signature:** `clip_grad_norm(parameters, max_norm) -> float`\n\n**Parameters:**\n- `parameters` — list of tensors with `.grad` attributes\n- `max_norm` — maximum allowed gradient norm\n\n**Returns:** original total gradient norm (float)\n\n**Constraints:**\n- Total norm = `sqrt(sum(p.grad.norm()^2))`\n- Only clip if total norm > max_norm\n- Preserve gradient direction",
    "description_zh": "实现梯度范数裁剪。\n\n梯度裁剪在所有参数梯度的 L2 范数超过阈值时进行缩放，防止梯度爆炸。\n\n**签名:** `clip_grad_norm(parameters, max_norm) -> float`\n\n**参数:**\n- `parameters` — 带 `.grad` 属性的张量列表\n- `max_norm` — 允许的最大梯度范数\n\n**返回:** 原始总梯度范数（浮点数）\n\n**约束:**\n- 总范数 = `sqrt(sum(p.grad.norm()^2))`\n- 仅在总范数 > max_norm 时裁剪\n- 保持梯度方向不变",
    "function_name": "clip_grad_norm",
    "hint": "1. total_norm = sqrt(Σ p.grad.norm()²)\n2. clip_coef = max_norm / total_norm\n3. if total_norm > max_norm: scale all grads by clip_coef\n4. return original total_norm (float)",
    "hint_zh": "1. total_norm = sqrt(Σ p.grad.norm()²)\n2. clip_coef = max_norm / total_norm\n3. 若 total_norm > max_norm：所有梯度乘以 clip_coef\n4. 返回原始 total_norm（float）",
    "tests": [
        {
            "name": "Clips to max_norm",
            "code": "\nimport torch\np1 = torch.randn(10, requires_grad=True)\np2 = torch.randn(10, requires_grad=True)\n(p1 * 10).sum().backward()\n(p2 * 10).sum().backward()\n{fn}([p1, p2], max_norm=1.0)\nnew_norm = torch.sqrt(p1.grad.norm()**2 + p2.grad.norm()**2).item()\nassert new_norm <= 1.0 + 1e-5, f'Clipped norm {new_norm:.4f} > 1.0'\n"
        },
        {
            "name": "Returns original norm",
            "code": "\nimport torch\np = torch.randn(10, requires_grad=True)\n(p * 3).sum().backward()\nexpected = p.grad.norm().item()\nreturned = {fn}([p], max_norm=100.0)\nassert abs(returned - expected) < 1e-4, f'Returned {returned:.4f}, expected {expected:.4f}'\n"
        },
        {
            "name": "No change when norm < max_norm",
            "code": "\nimport torch\np = torch.randn(4, requires_grad=True)\n(p * 0.001).sum().backward()\ngrad_before = p.grad.clone()\n{fn}([p], max_norm=100.0)\nassert torch.equal(p.grad, grad_before), 'Should not change when norm < max_norm'\n"
        },
        {
            "name": "Preserves direction",
            "code": "\nimport torch\ntorch.manual_seed(0)\np = torch.randn(100, requires_grad=True)\n(p * 10).sum().backward()\ndir_before = p.grad / p.grad.norm()\n{fn}([p], max_norm=1.0)\ndir_after = p.grad / p.grad.norm()\nassert torch.allclose(dir_before, dir_after, atol=1e-5), 'Should preserve direction'\n"
        }
    ],
    "solution": '''def clip_grad_norm(parameters, max_norm):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm.item()''',
    "demo": """p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
print('Before:', p.grad.norm().item())
orig = clip_grad_norm([p], max_norm=1.0)
print('After: ', p.grad.norm().item())
print('Returned:', orig)""",

}