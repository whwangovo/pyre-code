"""Speculative Decoding task."""

TASK = {
    "title": "Speculative Decoding",
    "title_zh": "推测解码",
    "difficulty": "Hard",
    "description_en": "Implement speculative decoding for faster LLM inference.\n\nSpeculative decoding uses a fast draft model to propose tokens, then verifies them against the target model, accepting or resampling based on probability ratios.\n\n**Signature:** `speculative_decode(target_probs, draft_probs, draft_tokens) -> list[int]`\n\n**Parameters:**\n- `target_probs` — target model probabilities (K, V)\n- `draft_probs` — draft model probabilities (K, V)\n- `draft_tokens` — proposed token IDs (K,)\n\n**Returns:** list of accepted token IDs (length 1 to K)\n\n**Constraints:**\n- Accept with prob `min(1, p_target/p_draft)`\n- On rejection, resample from `max(0, p_target - p_draft)` normalized\n- Stop at first rejection (return accepted so far + resampled token)",
    "description_zh": "实现推测解码以加速 LLM 推理。\n\n推测解码使用快速草稿模型提议 token，然后根据概率比率与目标模型验证，接受或重新采样。\n\n**签名:** `speculative_decode(target_probs, draft_probs, draft_tokens) -> list[int]`\n\n**参数:**\n- `target_probs` — 目标模型概率 (K, V)\n- `draft_probs` — 草稿模型概率 (K, V)\n- `draft_tokens` — 提议的 token ID (K,)\n\n**返回:** 接受的 token ID 列表（长度 1 到 K）\n\n**约束:**\n- 以概率 `min(1, p_target/p_draft)` 接受\n- 拒绝时从归一化的 `max(0, p_target - p_draft)` 重新采样\n- 在首次拒绝时停止（返回已接受的 + 重采样 token）",
    "function_name": "speculative_decode",
    "hint": "for i in range(K):\n  accept_prob = min(1, p_target[i, token] / p_draft[i, token])\n  if accepted: append token, continue\n  else: resample from max(0, p_target[i] - p_draft[i]) normalized, append, return",
    "hint_zh": "for i in range(K):\n  accept_prob = min(1, p_target[i, token] / p_draft[i, token])\n  若接受：追加 token，继续\n  若拒绝：从归一化的 max(0, p_target[i] - p_draft[i]) 重采样，追加后返回",
    "tests": [
        {
            "name": "Perfect draft: all accepted",
            "code": "\nimport torch\ntorch.manual_seed(0)\nprobs = torch.softmax(torch.randn(4, 10), dim=-1)\ntokens = torch.tensor([2, 5, 1, 8])\naccepted = {fn}(probs, probs, tokens)\nassert len(accepted) == 4, f'Perfect draft should accept all, got {len(accepted)}'\nfor i in range(4):\n    assert accepted[i] == tokens[i].item(), f'Token {i} mismatch'\n"
        },
        {
            "name": "Output length bounded",
            "code": "\nimport torch\ntorch.manual_seed(0)\nK = 5\ntarget = torch.softmax(torch.randn(K, 8), dim=-1)\ndraft = torch.softmax(torch.randn(K, 8), dim=-1)\ntokens = torch.randint(0, 8, (K,))\naccepted = {fn}(target, draft, tokens)\nassert 1 <= len(accepted) <= K, f'Length {len(accepted)} not in [1, {K}]'\n"
        },
        {
            "name": "All tokens valid",
            "code": "\nimport torch\nV = 8\nfor seed in range(20):\n    torch.manual_seed(seed)\n    target = torch.softmax(torch.randn(3, V), dim=-1)\n    draft = torch.softmax(torch.randn(3, V), dim=-1)\n    tokens = torch.randint(0, V, (3,))\n    for t in {fn}(target, draft, tokens):\n        assert 0 <= t < V, f'Token {t} out of range'\n"
        },
        {
            "name": "Rejection occurs when draft prob >> target prob",
            "code": "\nimport torch\n# Token 0: draft prob 0.9, target prob 0.1 -> acceptance ~11%\n# Run 100 times; rejection must occur at least once\ntarget_probs = torch.tensor([[0.1, 0.8, 0.1], [0.33, 0.33, 0.34]])\ndraft_probs  = torch.tensor([[0.9, 0.05, 0.05], [0.33, 0.33, 0.34]])\ndraft_tokens = torch.tensor([0, 1])\ntorch.manual_seed(99)\nrejections = 0\nfor _ in range(100):\n    accepted = {fn}(target_probs, draft_probs, draft_tokens)\n    if len(accepted) < 3:\n        rejections += 1\nassert rejections > 0, f'Token 0 should be rejected at least sometimes (acceptance prob ~11%), got 0 rejections in 100 trials'\n"
        }
    ],
    "solution": '''def speculative_decode(target_probs, draft_probs, draft_tokens):
    K = len(draft_tokens)
    accepted = []
    for i in range(K):
        t = draft_tokens[i].item()
        ratio = target_probs[i, t] / max(draft_probs[i, t].item(), 1e-10)
        if torch.rand(1).item() < min(1.0, ratio.item()):
            accepted.append(t)
        else:
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            s = adjusted.sum()
            if s > 0:
                adjusted = adjusted / s
            else:
                adjusted = torch.ones_like(adjusted) / adjusted.shape[0]
            accepted.append(torch.multinomial(adjusted, 1).item())
            return accepted
    return accepted''',
    "demo": """torch.manual_seed(0)
probs = torch.softmax(torch.randn(4, 10), dim=-1)
tokens = torch.tensor([2, 5, 1, 8])
print('Perfect draft:', speculative_decode(probs, probs, tokens))""",

}