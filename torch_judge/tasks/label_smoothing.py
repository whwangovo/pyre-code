"""Label Smoothing Loss task."""

TASK = {
    "title": "Label Smoothing Loss",
    "title_zh": "标签平滑损失",
    "difficulty": "Easy",
    "description_en": "Implement label smoothing cross-entropy loss.\n\nLabel smoothing prevents overconfidence by mixing the one-hot target distribution with a uniform distribution. It is widely used in Transformer training.\n\n**Signature:** `label_smoothing(logits, targets, smoothing=0.1) -> Tensor`\n\n**Parameters:**\n- `logits` — raw model outputs, shape `(N, C)`\n- `targets` — integer class indices, shape `(N,)`\n- `smoothing` — smoothing factor ε ∈ [0, 1)\n\n**Returns:** scalar loss\n\n**Formula:** soft target for correct class = `1 - ε`, for others = `ε / (C - 1)`",
    "description_zh": "实现标签平滑交叉熵损失。\n\n标签平滑通过将 one-hot 目标分布与均匀分布混合来防止过度自信，在 Transformer 训练中被广泛使用。\n\n**签名:** `label_smoothing(logits, targets, smoothing=0.1) -> Tensor`\n\n**参数:**\n- `logits` — 模型原始输出，形状 `(N, C)`\n- `targets` — 整数类别索引，形状 `(N,)`\n- `smoothing` — 平滑系数 ε ∈ [0, 1)\n\n**返回:** 标量损失\n\n**公式:** 正确类别的软目标 = `1 - ε`，其他类别 = `ε / (C - 1)`",
    "function_name": "label_smoothing",
    "hint": "1. `soft = full((N,C), ε/(C-1))`, then `soft.scatter_(1, targets, 1-ε)`\n2. `log_probs` via stable log-softmax\n3. `return -(soft * log_probs).sum(dim=-1).mean()`",
    "hint_zh": "1. `soft = full((N,C), ε/(C-1))`，再 `soft.scatter_(1, targets, 1-ε)`\n2. 用稳定 log-softmax 得到 `log_probs`\n3. `return -(soft * log_probs).sum(dim=-1).mean()`",
    "tests": [
        {
            "name": "Returns scalar",
            "code": """
import torch
logits = torch.randn(4, 10)
targets = torch.tensor([0, 3, 7, 9])
loss = {fn}(logits, targets)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'
""",
        },
        {
            "name": "smoothing=0 matches standard cross-entropy",
            "code": """
import torch
torch.manual_seed(42)
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))
loss_smooth = {fn}(logits, targets, smoothing=0.0)
loss_ce = torch.nn.functional.cross_entropy(logits, targets)
assert torch.allclose(loss_smooth, loss_ce, atol=1e-5), f'With smoothing=0, should match CE: {loss_smooth} vs {loss_ce}'
""",
        },
        {
            "name": "smoothing reduces confidence",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.randn(16, 8)
targets = torch.randint(0, 8, (16,))
loss_0 = {fn}(logits, targets, smoothing=0.0)
loss_s = {fn}(logits, targets, smoothing=0.1)
# Label smoothing increases loss when model is confident
assert loss_s.item() != loss_0.item(), 'Smoothed loss should differ from standard CE'
assert loss_s.item() > 0, 'Loss must be positive'
""",
        },
        {
            "name": "Gradient flows",
            "code": """
import torch
logits = torch.randn(4, 6, requires_grad=True)
targets = torch.tensor([0, 1, 2, 3])
loss = {fn}(logits, targets, smoothing=0.1)
loss.backward()
assert logits.grad is not None, 'Gradient not computed'
""",
        },
        {
            "name": "Exact smoothed loss value",
            "code": """
import torch
torch.manual_seed(0)
B, C = 4, 5
logits = torch.randn(B, C)
targets = torch.randint(0, C, (B,))
smoothing = 0.1
# Reference using solution formula: non-target = smoothing/(C-1), target = 1-smoothing
logits_max = logits.max(dim=-1, keepdim=True).values
shifted = logits - logits_max
log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
smooth_labels = torch.full((B, C), smoothing / (C - 1))
smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
expected = -(smooth_labels * log_probs).sum(dim=-1).mean()
out = {fn}(logits, targets, smoothing)
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected.item():.6f}, got {out.item():.6f}'
""",
        },
    ],
    "solution": '''def label_smoothing(logits, targets, smoothing=0.1):
    N, C = logits.shape
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    soft_targets = torch.full_like(log_probs, smoothing / (C - 1))
    soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return -(soft_targets * log_probs).sum(dim=-1).mean()''',
    "demo": """torch.manual_seed(0)
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))

loss_smoothed = label_smoothing(logits, targets, smoothing=0.1)
loss_ce = torch.nn.functional.cross_entropy(logits, targets)

print(f"Label smoothing loss (eps=0.1): {loss_smoothed.item():.4f}")
print(f"Standard CE loss (eps=0.0):    {loss_ce.item():.4f}")

loss_no_smooth = label_smoothing(logits, targets, smoothing=0.0)
print(f"Label smoothing loss (eps=0.0): {loss_no_smooth.item():.4f}  (matches CE: {torch.allclose(loss_no_smooth, loss_ce)})")""",

}
