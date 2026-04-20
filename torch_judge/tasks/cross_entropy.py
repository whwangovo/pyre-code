"""Cross-Entropy Loss task."""

TASK = {
    "title": "Cross-Entropy Loss",
    "title_zh": "交叉熵损失",
    "difficulty": "Easy",
    "description_en": "Implement cross-entropy loss for classification.\n\nCross-entropy measures the difference between predicted logits and true class labels. It is the standard loss for classification tasks.\n\n**Signature:** `cross_entropy_loss(logits, targets) -> Tensor`\n\n**Parameters:**\n- `logits` — raw scores (B, C) where C is the number of classes\n- `targets` — ground-truth class indices (B,)\n\n**Returns:** scalar mean loss\n\n**Constraints:**\n- Must be numerically stable (handle large logits)\n- Use log-sum-exp trick for stability",
    "description_zh": "实现分类交叉熵损失。\n\n交叉熵衡量预测 logits 与真实类别标签之间的差异，是分类任务的标准损失函数。\n\n**签名:** `cross_entropy_loss(logits, targets) -> Tensor`\n\n**参数:**\n- `logits` — 原始分数 (B, C)，C 为类别数\n- `targets` — 真实类别索引 (B,)\n\n**返回:** 标量平均损失\n\n**约束:**\n- 必须数值稳定（处理大 logits）\n- 使用 log-sum-exp 技巧保证稳定性",
    "function_name": "cross_entropy_loss",
    "hint": "1. `log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)`\n2. `return -log_probs[arange(B), targets].mean()`",
    "hint_zh": "1. `log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)`\n2. `return -log_probs[arange(B), targets].mean()`",
    "tests": [
        {
            "name": "Matches F.cross_entropy",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.randn(4, 10)\ntargets = torch.randint(0, 10, (4,))\nout = {fn}(logits, targets)\nref = torch.nn.functional.cross_entropy(logits, targets)\nassert torch.allclose(out, ref, atol=1e-5), f'Mismatch: {out.item():.4f} vs {ref.item():.4f}'\n"
        },
        {
            "name": "Numerical stability",
            "code": "\nimport torch\nlogits = torch.tensor([[1000., 0., 0.], [0., 1000., 0.]])\ntargets = torch.tensor([0, 1])\nout = {fn}(logits, targets)\nassert not torch.isnan(out), 'NaN with large logits'\nassert not torch.isinf(out), 'Inf with large logits'\nassert out.item() < 0.01, 'Should be ~0 for confident correct predictions'\n"
        },
        {
            "name": "Scalar output",
            "code": "\nimport torch\nout = {fn}(torch.randn(8, 5), torch.randint(0, 5, (8,)))\nassert out.dim() == 0, 'Loss must be a scalar'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nlogits = torch.randn(8, 5, requires_grad=True)\ntargets = torch.randint(0, 5, (8,))\n{fn}(logits, targets).backward()\nassert logits.grad is not None, 'logits.grad is None'\n"
        }
    ],
    "solution": '''def cross_entropy_loss(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs[torch.arange(targets.shape[0]), targets].mean()''',
    "demo": """logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())""",

}