"""DPO (Direct Preference Optimization) Loss task."""

TASK = {
    "title": "DPO (Direct Preference Optimization) Loss",
    "title_zh": "DPO 损失",
    "difficulty": "Medium",
    "description_en": "Implement the DPO (Direct Preference Optimization) loss.\n\nDPO aligns language models with human preferences without reinforcement learning, using paired chosen/rejected log-probabilities.\n\n**Signature:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor`\n\n**Parameters:**\n- `policy_chosen_logps` — policy log-probs for chosen responses (B,)\n- `policy_rejected_logps` — policy log-probs for rejected responses (B,)\n- `ref_chosen_logps`, `ref_rejected_logps` — reference model log-probs (B,)\n- `beta` — temperature scaling factor\n\n**Returns:** scalar loss\n\n**Constraints:**\n- `L = -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))).mean()`",
    "description_zh": "实现 DPO（直接偏好优化）损失。\n\nDPO 无需强化学习即可将语言模型与人类偏好对齐，使用配对的选中/拒绝对数概率。\n\n**签名:** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor`\n\n**参数:**\n- `policy_chosen_logps` — 策略模型对选中回复的对数概率 (B,)\n- `policy_rejected_logps` — 策略模型对拒绝回复的对数概率 (B,)\n- `ref_chosen_logps`, `ref_rejected_logps` — 参考模型的对数概率 (B,)\n- `beta` — 温度缩放因子\n\n**返回:** 标量损失\n\n**约束:**\n- `L = -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))).mean()`",
    "function_name": "dpo_loss",
    "hint": "1. chosen_logratios = β·(π_chosen - ref_chosen)\n2. rejected_logratios = β·(π_rejected - ref_rejected)\n3. loss = -log(sigmoid(chosen_logratios - rejected_logratios)).mean()",
    "hint_zh": "1. chosen_logratios = β·(π_chosen - ref_chosen)\n2. rejected_logratios = β·(π_rejected - ref_rejected)\n3. loss = -log(sigmoid(chosen_logratios - rejected_logratios)).mean()",
    "tests": [
        {
            "name": "Easy pair: small loss",
            "code": "\nimport torch\nchosen = torch.tensor([0.0, 0.0])\nrejected = torch.tensor([-10.0, -10.0])\nref_c = torch.tensor([-1.0, -1.0])\nref_r = torch.tensor([-1.0, -1.0])\nloss = {fn}(chosen, rejected, ref_c, ref_r, beta=0.1)\nassert loss.dim() == 0, 'Must be scalar'\nassert loss.item() < 0.5, f'Easy pair loss too high: {loss.item():.4f}'\n"
        },
        {
            "name": "Hard pair: large loss",
            "code": "\nimport torch\nloss = {fn}(torch.tensor([-10.0]), torch.tensor([0.0]),\n            torch.tensor([-1.0]), torch.tensor([-1.0]), beta=0.1)\nassert loss.item() > 0.5, f'Hard pair loss too low: {loss.item():.4f}'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nc = torch.randn(4, requires_grad=True)\nr = torch.randn(4, requires_grad=True)\n{fn}(c, r, torch.randn(4), torch.randn(4)).backward()\nassert c.grad is not None and r.grad is not None, 'Missing gradients'\n"
        },
        {
            "name": "Mathematical correctness",
            "code": "\nimport torch\ntorch.manual_seed(0)\nc, r, rc, rr = torch.randn(3), torch.randn(3), torch.randn(3), torch.randn(3)\nbeta = 0.5\nloss = {fn}(c, r, rc, rr, beta=beta)\nref = -torch.log(torch.sigmoid(beta * ((c - rc) - (r - rr)))).mean()\nassert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Beta scaling",
            "code": "\nimport torch\ntorch.manual_seed(0)\nc, r, rc, rr = torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4)\nl1 = {fn}(c, r, rc, rr, beta=0.1)\nl2 = {fn}(c, r, rc, rr, beta=1.0)\nassert not torch.allclose(l1, l2), 'Different beta should give different loss'\n"
        }
    ],
    "solution": '''def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    diff = chosen_rewards - rejected_rewards
    return -torch.log(torch.sigmoid(diff)).mean()''',
    "demo": """chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-5.0, -5.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
print('Loss:', dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())""",

}