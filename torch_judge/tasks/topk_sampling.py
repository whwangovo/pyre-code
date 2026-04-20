"""Top-k / Top-p Sampling task."""

TASK = {
    "title": "Top-k / Top-p Sampling",
    "title_zh": "Top-k / Top-p 采样",
    "difficulty": "Medium",
    "description_en": "Implement top-k / top-p (nucleus) sampling for language model decoding.\n\nThese sampling strategies filter the vocabulary to high-probability tokens before sampling, balancing diversity and quality in text generation.\n\n**Signature:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`\n\n**Parameters:**\n- `logits` — raw logits over vocabulary (V,)\n- `top_k` — keep only top-k tokens (0 = disabled)\n- `top_p` — keep tokens with cumulative prob <= p (1.0 = disabled)\n- `temperature` — temperature scaling\n\n**Returns:** sampled token index (int)\n\n**Constraints:**\n- Apply temperature first, then top-k, then top-p\n- `top_k=1` must always return argmax",
    "description_zh": "实现语言模型解码的 top-k / top-p（核）采样。\n\n这些采样策略在采样前将词表过滤为高概率 token，在文本生成中平衡多样性和质量。\n\n**签名:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`\n\n**参数:**\n- `logits` — 词表上的原始 logits (V,)\n- `top_k` — 仅保留 top-k 个 token（0 = 禁用）\n- `top_p` — 保留累积概率 <= p 的 token（1.0 = 禁用）\n- `temperature` — 温度缩放\n\n**返回:** 采样的 token 索引（整数）\n\n**约束:**\n- 先应用温度，再 top-k，再 top-p\n- `top_k=1` 必须始终返回 argmax",
    "function_name": "sample_top_k_top_p",
    "hint": "1. logits /= temperature\n2. top-k: set logits below k-th largest to -inf\n3. top-p: sort desc → cumsum of softmax probs → mask where cumsum > p → set to -inf\n4. sample from softmax(logits)",
    "hint_zh": "1. logits /= temperature\n2. top-k：将低于第 k 大值的 logits 设为 -inf\n3. top-p：降序排列 → softmax 概率 cumsum → 遮蔽 cumsum > p 的部分 → 设为 -inf\n4. 从 softmax(logits) 中采样",
    "tests": [
        {
            "name": "top_k=1 always returns argmax",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.tensor([1.0, 5.0, 2.0, 0.5])\nfor _ in range(10):\n    assert {fn}(logits.clone(), top_k=1) == 1, 'top_k=1 should return argmax'\n"
        },
        {
            "name": "Low temperature concentrates",
            "code": "\nimport torch\ntorch.manual_seed(42)\nlogits = torch.tensor([1.0, 3.0, 2.0])\ncounts = [0, 0, 0]\nfor _ in range(100):\n    counts[{fn}(logits.clone(), temperature=0.01)] += 1\nassert counts[1] > 90, f'Low temp should pick argmax, got {counts}'\n"
        },
        {
            "name": "All tokens reachable (no filtering)",
            "code": "\nimport torch\nlogits = torch.zeros(5)\nseen = set()\nfor i in range(200):\n    torch.manual_seed(i)\n    seen.add({fn}(logits.clone()))\nassert len(seen) == 5, f'Only saw {seen}'\n"
        },
        {
            "name": "Returns valid index",
            "code": "\nimport torch\ntorch.manual_seed(0)\nV = 100\nlogits = torch.randn(V)\nfor _ in range(20):\n    t = {fn}(logits.clone(), top_k=10, top_p=0.9)\n    assert 0 <= t < V, f'Token {t} out of range'\n"
        },
        {
            "name": "top_p excludes low-probability tokens",
            "code": "\nimport torch\n# Token 0 has prob ~1.0 after softmax; top_p=0.5 should only allow token 0\nlogits = torch.zeros(10)\nlogits[0] = 10.0\ntorch.manual_seed(0)\nresults = set()\nfor _ in range(50):\n    results.add({fn}(logits.clone(), top_k=0, top_p=0.5, temperature=1.0))\nexpected = {0}\nassert results == expected, f'With top_p=0.5 and dominant token 0, only token 0 should be sampled, got ' + str(results)\n"
        }
    ],
    "solution": '''def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        top_k_val = logits.topk(top_k).values[-1]
        logits[logits < top_k_val] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = (cumsum - probs) > top_p
        sorted_logits[mask] = float('-inf')
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()''',
    "demo": """logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
print('top_k=1:', sample_top_k_top_p(logits.clone(), top_k=1))
print('top_p=0.5:', sample_top_k_top_p(logits.clone(), top_p=0.5))""",

}