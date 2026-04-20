"""Focal Loss task."""

TASK = {
    "title": "Focal Loss",
    "title_zh": "Focal Loss",
    "difficulty": "Medium",
    "description_en": "Implement Focal Loss for handling class imbalance in classification.\n\nFocal Loss down-weights easy examples so the model focuses on hard ones. It was introduced in RetinaNet for dense object detection.\n\n**Signature:** `focal_loss(logits, targets, alpha=0.25, gamma=2.0) -> Tensor`\n\n**Parameters:**\n- `logits` — raw class scores, shape (N, C)\n- `targets` — integer class indices, shape (N,)\n- `alpha` — weighting factor (scalar)\n- `gamma` — focusing parameter; higher values down-weight easy examples more\n\n**Returns:** scalar mean loss\n\n**Formula:**\n```\nFL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)\n```\nwhere `p_t = softmax(logits)[i, targets[i]]`\n\n**Constraints:**\n- Compute softmax manually (no `F.*`)\n- Use numerically stable softmax (subtract max before exp)\n- Return the mean over the batch",
    "description_zh": "实现用于处理分类任务中类别不平衡问题的 Focal Loss。\n\nFocal Loss 降低简单样本的权重，使模型专注于困难样本。它由 RetinaNet 在密集目标检测中提出。\n\n**签名:** `focal_loss(logits, targets, alpha=0.25, gamma=2.0) -> Tensor`\n\n**参数:**\n- `logits` — 原始类别分数，形状 (N, C)\n- `targets` — 整数类别索引，形状 (N,)\n- `alpha` — 加权因子（标量）\n- `gamma` — 聚焦参数；值越大，对简单样本的降权越强\n\n**返回:** 标量均值损失\n\n**公式:**\n```\nFL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)\n```\n其中 `p_t = softmax(logits)[i, targets[i]]`\n\n**约束:**\n- 手动计算 softmax（不得使用 `F.*`）\n- 使用数值稳定的 softmax（exp 前减去最大值）\n- 返回批次上的均值",
    "function_name": "focal_loss",
    "hint": "1. Stable softmax: `exp(logits - max)` → normalize → `probs`\n2. `p_t = probs[arange(N), targets]`\n3. `return (-alpha * (1-p_t)**gamma * log(p_t+1e-8)).mean()`",
    "hint_zh": "1. 稳定 softmax：`exp(logits - max)` → 归一化 → `probs`\n2. `p_t = probs[arange(N), targets]`\n3. `return (-alpha * (1-p_t)**gamma * log(p_t+1e-8)).mean()`",
    "tests": [
        {
            "name": "Returns scalar",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.randn(8, 5)\ntargets = torch.randint(0, 5, (8,))\nloss = {fn}(logits, targets)\nassert loss.shape == (), f'Expected scalar, got shape {loss.shape}'\nassert loss.item() > 0, 'Loss should be positive'\n"
        },
        {
            "name": "gamma=0 reduces to weighted cross-entropy",
            "code": "\nimport torch\ntorch.manual_seed(1)\nlogits = torch.randn(16, 4)\ntargets = torch.randint(0, 4, (16,))\nalpha = 0.25\nloss_focal = {fn}(logits, targets, alpha=alpha, gamma=0.0)\n# reference: manual cross-entropy weighted by alpha\nN = logits.shape[0]\nshifted = logits - logits.max(dim=-1, keepdim=True).values\nprobs = torch.exp(shifted) / torch.exp(shifted).sum(dim=-1, keepdim=True)\np_t = probs[torch.arange(N), targets]\nce = (-alpha * torch.log(p_t + 1e-8)).mean()\nassert torch.allclose(loss_focal, ce, atol=1e-5), f'gamma=0 should equal weighted CE. Got {loss_focal.item():.6f} vs {ce.item():.6f}'\n"
        },
        {
            "name": "Higher gamma down-weights easy examples",
            "code": """
import torch
# easy example: p_t=0.9, hard example: p_t=0.3
# higher gamma should reduce the easy example's contribution more
torch.manual_seed(5)
logits = torch.randn(32, 4)
targets = torch.randint(0, 4, (32,))
loss_g0 = {fn}(logits, targets, alpha=1.0, gamma=0.0)
loss_g2 = {fn}(logits, targets, alpha=1.0, gamma=2.0)
loss_g5 = {fn}(logits, targets, alpha=1.0, gamma=5.0)
# higher gamma always reduces loss (down-weights easy examples)
assert loss_g5 <= loss_g2, f'gamma=5 ({loss_g5:.4f}) should be <= gamma=2 ({loss_g2:.4f})'
assert loss_g2 <= loss_g0, f'gamma=2 ({loss_g2:.4f}) should be <= gamma=0 ({loss_g0:.4f})'
""",
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nlogits = torch.randn(4, 3, requires_grad=True)\ntargets = torch.randint(0, 3, (4,))\nloss = {fn}(logits, targets)\nloss.backward()\nassert logits.grad is not None, 'Gradient did not flow back to logits'\nassert logits.grad.shape == logits.shape\n"
        }
    ],
    "solution": '''def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    N, C = logits.shape
    # numerically stable softmax
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exp_s = torch.exp(shifted)
    probs = exp_s / exp_s.sum(dim=-1, keepdim=True)
    # p_t: probability assigned to the correct class
    p_t = probs[torch.arange(N), targets]
    log_p_t = torch.log(p_t + 1e-8)
    fl = -alpha * (1 - p_t) ** gamma * log_p_t
    return fl.mean()''',
    "demo": """torch.manual_seed(0)
logits = torch.randn(8, 4)
targets = torch.randint(0, 4, (8,))

fl_gamma0 = focal_loss(logits, targets, alpha=0.25, gamma=0.0)
ce = F.cross_entropy(logits, targets)
print(f"Focal (gamma=0): {fl_gamma0:.4f}  |  alpha * CE: {0.25 * ce:.4f}")

fl_g2 = focal_loss(logits, targets, alpha=0.25, gamma=2.0)
fl_g5 = focal_loss(logits, targets, alpha=0.25, gamma=5.0)
print(f"Focal gamma=2: {fl_g2:.4f}  |  gamma=5: {fl_g5:.4f}  (higher gamma -> lower loss)")""",

}
