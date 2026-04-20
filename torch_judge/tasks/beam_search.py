"""Beam Search Decoding task."""

TASK = {
    "title": "Beam Search Decoding",
    "title_zh": "束搜索解码",
    "difficulty": "Medium",
    "description_en": "Implement beam search decoding for sequence generation.\n\nBeam search maintains multiple candidate sequences (beams) at each step, expanding and pruning to find the highest-scoring sequence.\n\n**Signature:** `beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list[int]`\n\n**Parameters:**\n- `log_prob_fn` — callable that takes a token sequence tensor and returns log-probabilities over vocabulary\n- `start_token` — integer start token\n- `beam_width` — number of beams to keep\n- `eos_token` — end-of-sequence token\n\n**Returns:** list of token IDs for the best sequence\n\n**Constraints:**\n- Stop when all beams end with eos or max_len is reached\n- Return the highest-scoring complete sequence",
    "description_zh": "实现序列生成的束搜索解码。\n\n束搜索在每一步维护多个候选序列（束），通过扩展和剪枝找到得分最高的序列。\n\n**签名:** `beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list[int]`\n\n**参数:**\n- `log_prob_fn` — 接受 token 序列张量并返回词表上对数概率的可调用对象\n- `start_token` — 起始 token 整数\n- `beam_width` — 保留的束数量\n- `eos_token` — 序列结束 token\n\n**返回:** 最佳序列的 token ID 列表\n\n**约束:**\n- 当所有束以 eos 结尾或达到 max_len 时停止\n- 返回得分最高的完整序列",
    "function_name": "beam_search",
    "hint": "1. beams = [(score=0.0, seq=[start_token])]\n2. each step: expand each beam with all tokens → keep top beam_width by cumulative score\n3. stop when all beams end with eos_token or max_len reached\n4. return seq from highest-scoring beam",
    "hint_zh": "1. beams = [(score=0.0, seq=[start_token])]\n2. 每步：用所有 token 扩展每个候选 → 按累积分保留前 beam_width 个\n3. 所有候选以 eos_token 结尾或达到 max_len 时停止\n4. 返回得分最高的序列",
    "tests": [
        {
            "name": "Returns list starting with start_token",
            "code": "\nimport torch\ndef dummy(tokens): return torch.zeros(10)\nseq = {fn}(dummy, start_token=0, max_len=5, beam_width=3, eos_token=9)\nassert isinstance(seq, list), 'Must return a list'\nassert seq[0] == 0, f'First token: {seq[0]}'\n"
        },
        {
            "name": "Greedy path (beam=1)",
            "code": "\nimport torch\ndef greedy_fn(tokens):\n    lp = torch.full((5,), -10.0)\n    lp[min(len(tokens), 4)] = 0.0\n    return lp\nseq = {fn}(greedy_fn, start_token=0, max_len=5, beam_width=1, eos_token=4)\nassert seq == [0, 1, 2, 3, 4], f'Greedy: {seq}'\n"
        },
        {
            "name": "Beam finds better path than greedy",
            "code": "\nimport torch\ndef tricky(tokens):\n    lp = torch.full((6,), -100.0)\n    if len(tokens) == 1:\n        lp[1] = -1.0; lp[2] = -0.5\n    elif tokens[-1] == 1:\n        lp[5] = 0.0\n    elif tokens[-1] == 2:\n        lp[5] = -10.0\n    else:\n        lp[5] = 0.0\n    return lp\nseq = {fn}(tricky, start_token=0, max_len=5, beam_width=2, eos_token=5)\nassert seq == [0, 1, 5], f'Beam should find [0,1,5], got {seq}'\n"
        },
        {
            "name": "Stops at eos",
            "code": "\nimport torch\ndef eos_fn(tokens):\n    lp = torch.zeros(4); lp[3] = 10.0; return lp\nseq = {fn}(eos_fn, start_token=0, max_len=100, beam_width=2, eos_token=3)\nassert seq[-1] == 3 and len(seq) == 2, f'Should be [0,3], got {seq}'\n"
        }
    ],
    "solution": '''def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    beams = [(0.0, [start_token])]
    completed = []
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                completed.append((score, seq))
                continue
            log_probs = log_prob_fn(torch.tensor(seq))
            topk_lp, topk_idx = log_probs.topk(beam_width)
            for j in range(beam_width):
                candidates.append((score + topk_lp[j].item(), seq + [topk_idx[j].item()]))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
    all_seqs = completed + beams
    all_seqs.sort(key=lambda x: x[0], reverse=True)
    return all_seqs[0][1]''',
    "demo": """def simple_fn(tokens):
    lp = torch.full((5,), -10.0)
    lp[min(len(tokens), 4)] = 0.0
    return lp
seq = beam_search(simple_fn, start_token=0, max_len=5, beam_width=2, eos_token=4)
print('Sequence:', seq)""",

}