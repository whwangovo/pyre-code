"""GRPO (Group Relative Policy Optimization) Loss task."""

TASK = {
    "title": "GRPO (Group Relative Policy Optimization) Loss",
    "title_zh": "GRPO 损失",
    "difficulty": "Hard",
    "description_en": "Implement the GRPO (Group Relative Policy Optimization) loss.\n\nGRPO normalizes rewards within each prompt group to compute advantages, then optimizes the policy using these group-relative advantages.\n\n**Signature:** `grpo_loss(logps, rewards, group_ids, eps=1e-5) -> Tensor`\n\n**Parameters:**\n- `logps` — policy log-probabilities (B,)\n- `rewards` — scalar rewards (B,)\n- `group_ids` — integer group identifiers (B,)\n- `eps` — epsilon for numerical stability\n\n**Returns:** scalar loss\n\n**Constraints:**\n- Per-group z-score normalization: `A_i = (r_i - mean_g) / (std_g + eps)`\n- Advantages must be detached (gradients flow only through logps)\n- Loss = `-mean(A_i * logps_i)`",
    "description_zh": "实现 GRPO（组相对策略优化）损失。\n\nGRPO 在每个提示组内归一化奖励以计算优势值，然后使用这些组相对优势优化策略。\n\n**签名:** `grpo_loss(logps, rewards, group_ids, eps=1e-5) -> Tensor`\n\n**参数:**\n- `logps` — 策略对数概率 (B,)\n- `rewards` — 标量奖励 (B,)\n- `group_ids` — 整数组标识符 (B,)\n- `eps` — 数值稳定性的 epsilon\n\n**返回:** 标量损失\n\n**约束:**\n- 组内 z-score 归一化：`A_i = (r_i - mean_g) / (std_g + eps)`\n- 优势值必须 detach（梯度仅通过 logps 流动）\n- 损失 = `-mean(A_i * logps_i)`",
    "function_name": "grpo_loss",
    "hint": (
        "1. for each group g: A_i = (r_i - mean_g) / (std_g + eps)\n"
        "2. advantages = A.detach()  (no grad through rewards)\n"
        "3. loss = -(advantages * logps).mean()"
    ),
    "hint_zh": "1. 对每个组 g：A_i = (r_i - mean_g) / (std_g + eps)\n2. advantages = A.detach()（梯度不流经奖励）\n3. loss = -(advantages * logps).mean()",
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "logps = torch.randn(6, requires_grad=True)\n"
            "rewards = torch.randn(6)\n"
            "group_ids = torch.tensor([0, 0, 0, 1, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "\n"
            "def _reference_grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor, eps: float = 1e-5) -> Tensor:\n"
            "    # Same semantics as the reference solution: per-group z-score then -E[A_i * logp_i].\n"
            "    logps = logps.view(-1)\n"
            "    rewards = rewards.view(-1)\n"
            "    group_ids = group_ids.view(-1)\n"
            "    unique_ids = group_ids.unique()\n"
            "    advantages = torch.empty_like(rewards)\n"
            "    for gid in unique_ids:\n"
            "        mask = group_ids == gid\n"
            "        r_g = rewards[mask]\n"
            "        mean_g = r_g.mean()\n"
            "        std_g = r_g.std(unbiased=False)\n"
            "        advantages[mask] = (r_g - mean_g) / (std_g + eps)\n"
            "    advantages_detached = advantages.detach()\n"
            "    return -(advantages_detached * logps).mean()\n"
            "\n"
            "logps = torch.tensor([0.0, -0.5, -1.0, -1.5])\n"
            "rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss_student = {fn}(logps, rewards, group_ids)\n"
            "loss_ref = _reference_grpo_loss(logps, rewards, group_ids)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), 'Loss should match reference implementation numerically on a fixed example'\n"
        },
        {
            "name": "Gradient flows to logps only",
            "code": "\n"
            "import torch\n"
            "logps = torch.randn(4, requires_grad=True)\n"
            "rewards = torch.randn(4, requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "assert logps.grad is not None and rewards.grad is None, 'Gradients should flow only through logps'\n"
        },
        {
            "name": "Group-wise normalization",
            "code": "\n"
            "import torch\n"
            "logps = torch.zeros(4, requires_grad=True)\n"
            "rewards = torch.tensor([0.0, 1.0, 10.0, 11.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "# Since each group has rewards [0,1] and [10,11], the normalized advantages\n"
            "# should be identical across groups, leading to identical gradients per position.\n"
            "assert torch.allclose(logps.grad[:2], logps.grad[2:]), 'Groups should be treated independently but symmetrically'\n"
        },
    ],
    "solution": '''def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:
    """Group Relative Policy Optimization (GRPO) loss.

    logps: (B,) policy log-probs for each sampled response
    rewards: (B,) scalar rewards for each response
    group_ids: (B,) integers, same id = same prompt/group
    returns: scalar loss (Tensor)
    """
    # Compute per-group normalized advantages A_i
    unique_ids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    for gid in unique_ids:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = r_g.mean()
        std_g = r_g.std(unbiased=False)
        advantages[mask] = (r_g - mean_g) / (std_g + eps)

    # Stop gradient through advantages
    advantages_detached = advantages.detach()

    # GRPO objective: -E[A_i * logpi_i]
    return -(advantages_detached * logps).mean()''',
    "demo": """logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
print('Loss:', grpo_loss(logps, rewards, group_ids).item())""",

}