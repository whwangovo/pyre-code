"""PPO (Proximal Policy Optimization) clipped loss task."""

TASK = {
    "title": "PPO (Proximal Policy Optimization) Clipped Loss",
    "title_zh": "PPO 损失",
    "difficulty": "Medium",
    "description_en": "Implement the PPO clipped surrogate loss.\n\nPPO constrains policy updates by clipping the importance sampling ratio, preventing destructively large updates during reinforcement learning.\n\n**Signature:** `ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2) -> Tensor`\n\n**Parameters:**\n- `new_logps` — current policy log-probs (B,)\n- `old_logps` — old policy log-probs (B,), treated as constant\n- `advantages` — advantage estimates (B,), treated as constant\n- `clip_ratio` — clipping range epsilon\n\n**Returns:** scalar loss\n\n**Constraints:**\n- Ratio: `r = exp(new_logps - old_logps.detach())`\n- Loss: `-min(r * A, clamp(r, 1-eps, 1+eps) * A).mean()`\n- Gradients flow only through new_logps",
    "description_zh": "实现 PPO 裁剪代理损失。\n\nPPO 通过裁剪重要性采样比率来约束策略更新，防止强化学习中的破坏性大幅更新。\n\n**签名:** `ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2) -> Tensor`\n\n**参数:**\n- `new_logps` — 当前策略对数概率 (B,)\n- `old_logps` — 旧策略对数概率 (B,)，视为常量\n- `advantages` — 优势估计 (B,)，视为常量\n- `clip_ratio` — 裁剪范围 epsilon\n\n**返回:** 标量损失\n\n**约束:**\n- 比率：`r = exp(new_logps - old_logps.detach())`\n- 损失：`-min(r * A, clamp(r, 1-eps, 1+eps) * A).mean()`\n- 梯度仅通过 new_logps 流动",
    "function_name": "ppo_loss",
    "hint": (
        "1. r = exp(new_logps - old_logps.detach())\n"
        "2. unclipped = r * advantages.detach()\n"
        "3. clipped = clamp(r, 1-ε, 1+ε) * advantages.detach()\n"
        "4. loss = -min(unclipped, clipped).mean()"
    ),
    "hint_zh": "1. r = exp(new_logps - old_logps.detach())\n2. unclipped = r * advantages.detach()\n3. clipped = clamp(r, 1-ε, 1+ε) * advantages.detach()\n4. loss = -min(unclipped, clipped).mean()",
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "new_logps = torch.randn(16, requires_grad=True)\n"
            "old_logps = torch.randn(16)\n"
            "advantages = torch.randn(16)\n"
            "loss = {fn}(new_logps, old_logps, advantages)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs fixed value",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])\n"
            "old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])\n"
            "advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])\n"
            "loss = {fn}(new_logps, old_logps, advantages, clip_ratio=0.2)\n"
            "expected = torch.tensor(-0.0488)\n"
            "assert torch.allclose(loss, expected, atol=1e-4, rtol=0), 'Loss should match the expected numeric value on the fixed example'\n"
        },
        {
            "name": "Gradient flows to new_logps only",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.randn(8, requires_grad=True)\n"
            "old_logps = torch.randn(8, requires_grad=True)\n"
            "advantages = torch.randn(8, requires_grad=True)\n"
            "loss = {fn}(new_logps, old_logps, advantages)\n"
            "loss.backward()\n"
            "assert new_logps.grad is not None, 'Gradients should flow through new_logps'\n"
            "assert old_logps.grad is None, 'Gradients should not flow through old_logps (treat as constant baseline)'\n"
            "assert advantages.grad is None, 'Gradients should not flow through advantages (treat as constant advantages)'\n"
        },
    ],
    "solution": '''def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    """PPO clipped surrogate loss.

    new_logps: (B,) current policy log-probs
    old_logps: (B,) old policy log-probs (treated as constant)
    advantages: (B,) advantage estimates (treated as constant)
    returns: scalar loss (Tensor)
    """
    # Detach old_logps and advantages so gradients only flow through new_logps
    old_logps_detached = old_logps.detach()
    adv_detached = advantages.detach()

    # Importance sampling ratio r = pi_new / pi_old in log-space
    ratios = torch.exp(new_logps - old_logps_detached)

    # Unclipped and clipped objectives
    unclipped = ratios * adv_detached
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_detached

    # PPO objective: negative mean of the more conservative objective
    return -torch.min(unclipped, clipped).mean()''',
    "demo": """new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])
old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
print('Loss:', ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2))""",

}