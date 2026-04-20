"""MCTS for Reasoning task."""

TASK = {
    "title": "MCTS for Reasoning",
    "title_zh": "推理蒙特卡洛树搜索",
    "difficulty": "Hard",
    "description_en": "Implement one step of Monte Carlo Tree Search (MCTS) — the core algorithm behind o1/AlphaProof-style LLM reasoning.\n\n**Signature:** `mcts_step(values, visit_counts, parent_visits, c_puct=1.414) -> (int, Tensor, Tensor)`\n\n**Parameters:**\n- `values` — current value estimates for N child nodes (N,)\n- `visit_counts` — visit counts for each child (N,)\n- `parent_visits` — total visits to the parent node (int or scalar)\n- `c_puct` — exploration constant (default 1.414)\n\n**Returns:** `(selected_idx, updated_values, updated_visits)`\n\n**Algorithm:**\n1. **Selection** — compute UCB1 scores and pick `argmax`:\n   `UCB(i) = Q(i) + c_puct * sqrt(log(parent_visits + 1) / (visit_counts[i] + 1))`\n2. **Rollout** — simulate a random value in [0, 1] for the selected node\n3. **Backpropagation** — update with running mean:\n   `Q_new = (Q_old * n + rollout_value) / (n + 1)` where `n` is the old visit count\n4. Increment `visit_counts[selected_idx]` by 1",
    "description_zh": "实现蒙特卡洛树搜索（MCTS）的单步操作——o1/AlphaProof 风格 LLM 推理的核心算法。\n\n**签名:** `mcts_step(values, visit_counts, parent_visits, c_puct=1.414) -> (int, Tensor, Tensor)`\n\n**参数:**\n- `values` — N 个子节点的当前价值估计 (N,)\n- `visit_counts` — 每个子节点的访问次数 (N,)\n- `parent_visits` — 父节点的总访问次数（int 或标量）\n- `c_puct` — 探索常数（默认 1.414）\n\n**返回:** `(selected_idx, updated_values, updated_visits)`\n\n**算法:**\n1. **选择** — 计算 UCB1 分数并取 `argmax`：\n   `UCB(i) = Q(i) + c_puct * sqrt(log(parent_visits + 1) / (visit_counts[i] + 1))`\n2. **模拟** — 为选中节点模拟 [0, 1] 内的随机价值\n3. **反向传播** — 用滑动均值更新：\n   `Q_new = (Q_old * n + rollout_value) / (n + 1)`，其中 `n` 为旧访问次数\n4. 将 `visit_counts[selected_idx]` 加 1",
    "function_name": "mcts_step",
    "hint": "1. UCB(i) = values[i] + c_puct·√(log(parent_visits+1) / (visit_counts[i]+1))\n2. selected = argmax(UCB)\n3. rollout = torch.rand(1).item()  (random value in [0,1])\n4. Q_new = (Q_old·n + rollout) / (n+1);  visit_counts[selected] += 1",
    "hint_zh": "1. UCB(i) = values[i] + c_puct·√(log(parent_visits+1) / (visit_counts[i]+1))\n2. selected = argmax(UCB)\n3. rollout = torch.rand(1).item()（[0,1] 内随机值）\n4. Q_new = (Q_old·n + rollout) / (n+1);  visit_counts[selected] += 1",
    "tests": [
        {
            "name": "Return types and shapes",
            "code": "\nimport torch\ntorch.manual_seed(0)\nvalues = torch.rand(5)\nvisits = torch.zeros(5, dtype=torch.float32)\nidx, new_vals, new_visits = {fn}(values, visits, parent_visits=10)\nassert isinstance(idx, int), f'selected_idx must be int, got {type(idx)}'\nassert new_vals.shape == values.shape, f'values shape mismatch: {new_vals.shape}'\nassert new_visits.shape == visits.shape, f'visits shape mismatch: {new_visits.shape}'\n"
        },
        {
            "name": "Unvisited nodes preferred",
            "code": "\nimport torch\n# All nodes visited except node 2 — UCB should strongly prefer node 2\nvalues = torch.zeros(4)\nvisits = torch.tensor([10.0, 10.0, 0.0, 10.0])\nidx, _, _ = {fn}(values, visits, parent_visits=30)\nassert idx == 2, f'Expected unvisited node 2 to be selected, got {idx}'\n"
        },
        {
            "name": "Visit count incremented",
            "code": "\nimport torch\ntorch.manual_seed(42)\nvalues = torch.rand(6)\nvisits = torch.ones(6)\nidx, _, new_visits = {fn}(values, visits, parent_visits=6)\nassert new_visits[idx] == visits[idx] + 1, 'visit_counts[selected_idx] must increase by 1'\n# Other counts unchanged\nfor i in range(6):\n    if i != idx:\n        assert new_visits[i] == visits[i], f'visit_counts[{i}] should not change'\n"
        },
        {
            "name": "Value updated via running mean",
            "code": "\nimport torch\ntorch.manual_seed(7)\nvalues = torch.tensor([0.5, 0.5, 0.5])\nvisits = torch.tensor([2.0, 2.0, 2.0])\nidx, new_vals, _ = {fn}(values, visits, parent_visits=6)\n# new value must differ from old (running mean with random rollout)\nassert new_vals[idx] != values[idx], 'Value at selected_idx should be updated'\n# Other values unchanged\nfor i in range(3):\n    if i != idx:\n        assert new_vals[i] == values[i], f'values[{i}] should not change'\n"
        },
        {
            "name": "Pure exploitation with c_puct=0",
            "code": "\nimport torch\nvalues = torch.tensor([0.2, 0.9, 0.5, 0.1])\nvisits = torch.tensor([1.0, 1.0, 1.0, 1.0])\nidx, _, _ = {fn}(values, visits, parent_visits=4, c_puct=0.0)\nassert idx == 1, f'With c_puct=0, should select highest-value node (1), got {idx}'\n"
        }
    ],
    "solution": '''def mcts_step(values, visit_counts, parent_visits, c_puct=1.414):
    # UCB1 scores
    exploration = c_puct * torch.sqrt(
        torch.log(torch.tensor(parent_visits + 1, dtype=torch.float32)) /
        (visit_counts.float() + 1)
    )
    ucb = values + exploration
    selected_idx = int(ucb.argmax().item())
    # Simulate rollout: random value in [0, 1]
    rollout_value = torch.rand(1).item()
    n = visit_counts[selected_idx].item()
    new_value = (values[selected_idx].item() * n + rollout_value) / (n + 1)
    updated_values = values.clone()
    updated_visits = visit_counts.clone()
    updated_values[selected_idx] = new_value
    updated_visits[selected_idx] += 1
    return selected_idx, updated_values, updated_visits''',
    "demo": """torch.manual_seed(42)
num_children = 5
values = torch.zeros(num_children)
visits = torch.zeros(num_children, dtype=torch.long)
parent_visits = 0

print("Running 5 MCTS steps:")
print(f"{'Step':>4}  {'Selected':>8}  {'Visit counts'}")
for step in range(5):
    idx, values, visits = mcts_step(values, visits, parent_visits)
    parent_visits += 1
    print(f"{step+1:>4}  {idx:>8}  {visits.tolist()}")

print(f"\nTotal visits: {visits.sum().item()} (expected 5)")
print(f"All nodes visited at least once: {(visits > 0).all().item()}")""",

}
