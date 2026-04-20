"""MoE Load Balancing Loss task."""

TASK = {
    "title": "MoE Load Balancing Loss",
    "title_zh": "MoE 负载均衡损失",
    "difficulty": "Medium",
    "description_en": "Implement the auxiliary load balancing loss from Switch Transformer / DeepSeek-style Mixture-of-Experts.\n\nWithout this loss, MoE models collapse: all tokens route to the same expert, wasting capacity. The loss penalizes uneven token distribution by combining a non-differentiable routing fraction with a differentiable probability term.\n\n**Signature:** `moe_load_balance_loss(router_logits, num_experts) -> Tensor`\n\n**Parameters:**\n- `router_logits` — raw router scores before softmax (N_tokens, num_experts)\n- `num_experts` — number of experts\n\n**Returns:** scalar auxiliary loss\n\n**Formula:**\n- `f_i` = fraction of tokens assigned to expert i (via argmax — non-differentiable)\n- `P_i` = mean router probability for expert i (via softmax — differentiable)\n- `L_aux = num_experts * Σ(f_i * P_i)`\n\n**Note:** Gradient flows only through `P_i` (the softmax term). `f_i` is a stop-gradient counting term.",
    "description_zh": "实现 Switch Transformer / DeepSeek 风格混合专家（MoE）中的辅助负载均衡损失。\n\n没有这个损失，MoE 模型会退化：所有 token 都路由到同一个专家，浪费容量。该损失通过将不可微的路由分数与可微的概率项结合，惩罚不均匀的 token 分布。\n\n**签名:** `moe_load_balance_loss(router_logits, num_experts) -> Tensor`\n\n**参数:**\n- `router_logits` — softmax 前的原始路由分数 (N_tokens, num_experts)\n- `num_experts` — 专家数量\n\n**返回:** 标量辅助损失\n\n**公式:**\n- `f_i` = 路由到专家 i 的 token 比例（通过 argmax，不可微）\n- `P_i` = 专家 i 的平均路由概率（通过 softmax，可微）\n- `L_aux = num_experts * Σ(f_i * P_i)`\n\n**注意:** 梯度只通过 `P_i`（softmax 项）传播，`f_i` 是停止梯度的计数项。",
    "function_name": "moe_load_balance_loss",
    "hint": "1. `assignments = logits.argmax(dim=-1)` → `f[e] = (assignments == e).float().mean()`\n2. `P = softmax(logits, dim=-1).mean(dim=0)`\n3. return `num_experts * (f * P).sum()`",
    "hint_zh": "1. `assignments = logits.argmax(dim=-1)` → `f[e] = (assignments == e).float().mean()`\n2. `P = softmax(logits, dim=-1).mean(dim=0)`\n3. 返回 `num_experts * (f * P).sum()`",
    "tests": [
        {
            "name": "Returns scalar",
            "code": """
import torch
torch.manual_seed(0)
N_tokens, num_experts = 16, 4
logits = torch.randn(N_tokens, num_experts)
loss = {fn}(logits, num_experts)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'
assert loss.item() > 0, 'Loss should be positive'
"""
        },
        {
            "name": "Uniform routing gives loss = 1.0",
            "code": "\nimport torch\n# Explicitly construct uniform routing: each expert gets N/E tokens\nnum_experts = 4\nN_tokens = 100\nlogits = torch.zeros(N_tokens, num_experts)\nfor e in range(num_experts):\n    logits[e * 25:(e + 1) * 25, e] = 10.0\n# f_i = 1/4 for all i, P_i ≈ softmax-dominated by the assigned expert\n# With large logit gap, P_i ≈ 1/4 on average, so L_aux ≈ 4 * 4 * (1/4)^2 = 1.0\nloss = {fn}(logits, num_experts)\nassert abs(loss.item() - 1.0) < 0.05, f'Uniform routing should give loss≈1.0, got {loss.item():.4f}'\n"
        },
        {
            "name": "Collapsed routing gives loss > 1",
            "code": """
import torch
num_experts = 4
N_tokens = 20
# All tokens routed to expert 0 via very large logit
logits = torch.zeros(N_tokens, num_experts)
logits[:, 0] = 100.0
loss = {fn}(logits, num_experts)
assert loss.item() > 1.0, f'Collapsed routing should give loss > 1.0, got {loss.item():.4f}'
"""
        },
        {
            "name": "Gradient flows through router_logits",
            "code": """
import torch
torch.manual_seed(1)
N_tokens, num_experts = 12, 3
logits = torch.randn(N_tokens, num_experts, requires_grad=True)
loss = {fn}(logits, num_experts)
loss.backward()
assert logits.grad is not None, 'Missing gradient for router_logits'
assert logits.grad.shape == logits.shape, 'Gradient shape mismatch'
"""
        }
    ],
    "solution": '''def moe_load_balance_loss(router_logits, num_experts):
    N = router_logits.shape[0]
    # f_i: fraction of tokens routed to each expert (argmax, non-differentiable)
    assignments = router_logits.argmax(dim=-1)  # (N,)
    f = torch.zeros(num_experts, device=router_logits.device)
    for e in range(num_experts):
        f[e] = (assignments == e).float().mean()
    # P_i: mean router probability per expert (differentiable via softmax)
    probs = torch.softmax(router_logits, dim=-1)  # (N, num_experts)
    P = probs.mean(dim=0)                          # (num_experts,)
    return num_experts * (f * P).sum()''',
    "demo": """num_experts = 4
N_tokens = 100
logits_uniform = torch.zeros(N_tokens, num_experts)
loss_uniform = moe_load_balance_loss(logits_uniform, num_experts)
print(f"Uniform routing loss: {loss_uniform.item():.4f}  (expected 1.0)")

logits_collapsed = torch.zeros(N_tokens, num_experts)
logits_collapsed[:, 0] = 100.0
loss_collapsed = moe_load_balance_loss(logits_collapsed, num_experts)
print(f"Collapsed routing loss: {loss_collapsed.item():.4f}  (expected > 1.0)")

torch.manual_seed(0)
logits_grad = torch.randn(16, num_experts, requires_grad=True)
loss_grad = moe_load_balance_loss(logits_grad, num_experts)
loss_grad.backward()
print(f"Gradient exists: {logits_grad.grad is not None}")""",

}
