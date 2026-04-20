"""Multi-Token Prediction Loss task."""

TASK = {
    "title": "Multi-Token Prediction Loss",
    "title_zh": "多 Token 预测",
    "difficulty": "Medium",
    "description_en": "Implement the Multi-Token Prediction (MTP) training loss from Meta's 2024 paper.\n\nInstead of predicting only the next token, train N independent prediction heads to simultaneously predict the next N tokens from the same hidden states. Average the cross-entropy loss across all heads.\n\n**Signature:** `multi_token_prediction_loss(hidden_states, heads, targets) -> Tensor`\n\n**Parameters:**\n- `hidden_states` — transformer hidden states (B, S, D)\n- `heads` — list of N weight matrices, each (D, vocab_size)\n- `targets` — target token IDs (B, S, N), where `targets[:, :, i]` are the labels for head i\n\n**Returns:** scalar mean loss (average CE across all N heads)\n\n**Constraints:**\n- Implement cross-entropy manually: log-softmax then gather\n- Do NOT use F.cross_entropy or F.log_softmax\n- Use numerically stable log-softmax: subtract max before exp",
    "description_zh": "实现 Meta 2024 年论文中的多 Token 预测（MTP）训练损失。\n\n不只预测下一个 token，而是训练 N 个独立的预测头，从相同的隐藏状态同时预测接下来的 N 个 token。对所有头的交叉熵损失取平均。\n\n**签名:** `multi_token_prediction_loss(hidden_states, heads, targets) -> Tensor`\n\n**参数:**\n- `hidden_states` — Transformer 隐藏状态 (B, S, D)\n- `heads` — N 个权重矩阵的列表，每个形状为 (D, vocab_size)\n- `targets` — 目标 token ID (B, S, N)，其中 `targets[:, :, i]` 是第 i 个头的标签\n\n**返回:** 标量均值损失（所有 N 个头的 CE 均值）\n\n**约束:**\n- 手动实现交叉熵：log-softmax 后 gather\n- 不能使用 F.cross_entropy 或 F.log_softmax\n- 使用数值稳定的 log-softmax：exp 前先减去最大值",
    "function_name": "multi_token_prediction_loss",
    "hint": "For each head `i`:\n1. `logits = hidden_states @ heads[i]`\n2. Stable log-softmax: `shifted = logits - logits.max(-1,keepdim=True).values`\n   `log_probs = shifted - log(exp(shifted).sum(-1,keepdim=True))`\n3. `log_p = log_probs.gather(-1, targets[:,:,i].unsqueeze(-1)).squeeze(-1)`\n4. `loss += -log_p.mean()`\nReturn `total_loss / N`",
    "hint_zh": "对每个头 `i`：\n1. `logits = hidden_states @ heads[i]`\n2. 稳定 log-softmax：`shifted = logits - logits.max(-1,keepdim=True).values`\n   `log_probs = shifted - log(exp(shifted).sum(-1,keepdim=True))`\n3. `log_p = log_probs.gather(-1, targets[:,:,i].unsqueeze(-1)).squeeze(-1)`\n4. `loss += -log_p.mean()`\n返回 `total_loss / N`",
    "tests": [
        {
            "name": "Returns scalar",
            "code": """
import torch
torch.manual_seed(0)
B, S, D, vocab, N = 2, 6, 16, 50, 3
hidden = torch.randn(B, S, D)
heads = [torch.randn(D, vocab) for _ in range(N)]
targets = torch.randint(0, vocab, (B, S, N))
loss = {fn}(hidden, heads, targets)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'
assert loss.item() > 0, 'Loss should be positive'
"""
        },
        {
            "name": "N=1 matches standard next-token CE",
            "code": """
import torch
torch.manual_seed(1)
B, S, D, vocab = 1, 4, 8, 20
hidden = torch.randn(B, S, D)
W = torch.randn(D, vocab)
targets_3d = torch.randint(0, vocab, (B, S, 1))
loss_mtp = {fn}(hidden, [W], targets_3d)
# Manual CE for reference
logits = hidden @ W
lmax = logits.max(dim=-1, keepdim=True).values
shifted = logits - lmax
log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
log_p = log_probs.gather(-1, targets_3d[:, :, 0].unsqueeze(-1)).squeeze(-1)
expected = -log_p.mean()
assert torch.allclose(loss_mtp, expected, atol=1e-5), f'N=1 loss {loss_mtp.item():.4f} != expected {expected.item():.4f}'
"""
        },
        {
            "name": "Loss decreases when predictions are correct",
            "code": """
import torch
torch.manual_seed(2)
B, S, D, vocab, N = 1, 3, 8, 10, 2
hidden = torch.randn(B, S, D)
targets = torch.randint(0, vocab, (B, S, N))
# Random heads -> high loss
heads_random = [torch.randn(D, vocab) for _ in range(N)]
loss_random = {fn}(hidden, heads_random, targets)
# Heads tuned to produce correct logits -> low loss
heads_good = []
for i in range(N):
    W = torch.zeros(D, vocab)
    # Make the target class have high logit
    for b in range(B):
        for s in range(S):
            t = targets[b, s, i].item()
            W[:, t] += hidden[b, s] * 10
    heads_good.append(W)
loss_good = {fn}(hidden, heads_good, targets)
assert loss_good < loss_random, f'Good predictions should have lower loss: {loss_good.item():.4f} vs {loss_random.item():.4f}'
"""
        },
        {
            "name": "Gradient flows through hidden_states and heads",
            "code": """
import torch
torch.manual_seed(3)
B, S, D, vocab, N = 1, 4, 8, 15, 2
hidden = torch.randn(B, S, D, requires_grad=True)
heads = [torch.randn(D, vocab, requires_grad=True) for _ in range(N)]
targets = torch.randint(0, vocab, (B, S, N))
loss = {fn}(hidden, heads, targets)
loss.backward()
assert hidden.grad is not None, 'Missing gradient for hidden_states'
for i, h in enumerate(heads):
    assert h.grad is not None, f'Missing gradient for heads[{i}]'
"""
        }
    ],
    "solution": '''def multi_token_prediction_loss(hidden_states, heads, targets):
    B, S, D = hidden_states.shape
    N = len(heads)
    total_loss = 0.0
    for i, head in enumerate(heads):
        logits = hidden_states @ head          # (B, S, vocab_size)
        # Numerically stable log-softmax
        logits_max = logits.max(dim=-1, keepdim=True).values
        shifted = logits - logits_max
        log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
        # Gather log-prob of target tokens
        tgt = targets[:, :, i]                 # (B, S)
        log_p = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total_loss = total_loss + (-log_p.mean())
    return total_loss / N''',
    "demo": """torch.manual_seed(0)
B, S, D, V = 2, 5, 16, 10

hidden = torch.randn(B, S, D)
head_single = torch.randn(D, V)
targets_single = torch.randint(0, V, (B, S, 1))

mtp_loss = multi_token_prediction_loss(hidden, [head_single], targets_single)

logits = hidden @ head_single
ce_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), targets_single[:, :, 0].reshape(-1))

print(f"MTP loss (N=1):  {mtp_loss.item():.6f}")
print(f"Standard CE:     {ce_loss.item():.6f}")
print(f"N=1 matches CE:  {torch.allclose(mtp_loss, ce_loss, atol=1e-5)}")

heads_3 = [torch.randn(D, V) for _ in range(3)]
targets_3 = torch.randint(0, V, (B, S, 3))
loss_3 = multi_token_prediction_loss(hidden, heads_3, targets_3)
print(f"MTP loss (N=3):  {loss_3.item():.6f}")""",

}
