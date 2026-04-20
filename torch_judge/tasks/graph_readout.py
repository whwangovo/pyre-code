"""Graph Readout (Graph-Level Pooling) task."""

TASK = {
    "title": "Graph Readout (Graph-Level Pooling)",
    "title_zh": "图读出（图级池化）",
    "difficulty": "Easy",
    "description_en": "Implement graph-level readout (pooling) that aggregates node embeddings into a single vector per graph.\n\n**Signature:** `graph_readout(X, batch, mode='mean') -> Tensor`\n\n**Parameters:**\n- `X` — node embedding matrix (total_nodes, D)\n- `batch` — integer tensor (total_nodes,) mapping each node to its graph index (0-indexed)\n- `mode` — aggregation mode: `'mean'`, `'sum'`, or `'max'`\n\n**Returns:** graph-level embeddings of shape (num_graphs, D) where `num_graphs = batch.max() + 1`\n\n**Algorithm:**\n- `'sum'`: accumulate node features per graph (e.g., `index_add_` or `scatter_add_`)\n- `'mean'`: sum per graph divided by node count per graph\n- `'max'`: per-graph element-wise maximum over node features\n\n**Constraints:**\n- Do not use PyG or DGL — use only raw PyTorch operations\n- Handle all three modes",
    "description_zh": "实现图级别的读出（池化）操作，将节点嵌入聚合为每个图的单一向量表示。\n\n**签名:** `graph_readout(X, batch, mode='mean') -> Tensor`\n\n**参数:**\n- `X` — 节点嵌入矩阵 (total_nodes, D)\n- `batch` — 整数张量 (total_nodes,)，将每个节点映射到所属图的索引（从 0 开始）\n- `mode` — 聚合方式：`'mean'`、`'sum'` 或 `'max'`\n\n**返回:** 形状为 (num_graphs, D) 的图级嵌入，其中 `num_graphs = batch.max() + 1`\n\n**算法:**\n- `'sum'`：按图累加节点特征（可用 `index_add_` 或 `scatter_add_`）\n- `'mean'`：按图求和后除以各图节点数\n- `'max'`：按图对节点特征逐元素取最大值\n\n**约束:**\n- 仅使用原生 PyTorch 操作，不得使用 PyG 或 DGL\n- 需支持全部三种聚合模式",
    "function_name": "graph_readout",
    "hint": "1. `num_graphs = batch.max().item() + 1`\n2. For sum: `out.index_add_(0, batch, X)`\n3. For mean: sum / count per graph\n4. For max: loop over graph indices or use `scatter` with `reduce='amax'`",
    "hint_zh": "1. `num_graphs = batch.max().item() + 1`\n2. sum 模式：`out.index_add_(0, batch, X)`\n3. mean 模式：sum / 每个图的节点数\n4. max 模式：按图索引循环或使用 `scatter` 配合 `reduce='amax'`",
    "tests": [
        {
            "name": "Single graph (all batch=0), mean mode",
            "code": """
import torch
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
batch = torch.tensor([0, 0, 0])
out = {fn}(X, batch, mode='mean')
assert out.shape == (1, 2), f'Expected (1, 2), got {out.shape}'
expected = X.mean(dim=0, keepdim=True)
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'
""",
        },
        {
            "name": "Multi-graph batch, sum mode",
            "code": """
import torch
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
batch = torch.tensor([0, 0, 1, 1])
out = {fn}(X, batch, mode='sum')
assert out.shape == (2, 2), f'Expected (2, 2), got {out.shape}'
expected = torch.tensor([[4.0, 6.0], [12.0, 14.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'
""",
        },
        {
            "name": "Sum mode accumulates correctly",
            "code": """
import torch
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
batch = torch.tensor([0, 1, 0])
out = {fn}(X, batch, mode='sum')
assert out.shape == (2, 2), f'Expected (2, 2), got {out.shape}'
expected = torch.tensor([[3.0, 3.0], [0.0, 1.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'
""",
        },
        {
            "name": "Mean mode divides by node count",
            "code": """
import torch
X = torch.tensor([[2.0, 4.0], [4.0, 6.0], [10.0, 20.0]])
batch = torch.tensor([0, 0, 1])
out = {fn}(X, batch, mode='mean')
expected = torch.tensor([[3.0, 5.0], [10.0, 20.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'
""",
        },
        {
            "name": "Max mode picks element-wise maximum",
            "code": """
import torch
X = torch.tensor([[1.0, 5.0], [3.0, 2.0], [4.0, 1.0]])
batch = torch.tensor([0, 0, 1])
out = {fn}(X, batch, mode='max')
expected = torch.tensor([[3.0, 5.0], [4.0, 1.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'
""",
        },
        {
            "name": "Exact numerical value with three graphs",
            "code": """
import torch
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
batch = torch.tensor([0, 0, 1, 2, 2])
out_sum = {fn}(X, batch, mode='sum')
out_mean = {fn}(X, batch, mode='mean')
out_max = {fn}(X, batch, mode='max')
exp_sum = torch.tensor([[4.0, 6.0], [5.0, 6.0], [16.0, 18.0]])
exp_mean = torch.tensor([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])
exp_max = torch.tensor([[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]])
assert torch.allclose(out_sum, exp_sum, atol=1e-5), f'Sum mismatch: {out_sum} vs {exp_sum}'
assert torch.allclose(out_mean, exp_mean, atol=1e-5), f'Mean mismatch: {out_mean} vs {exp_mean}'
assert torch.allclose(out_max, exp_max, atol=1e-5), f'Max mismatch: {out_max} vs {exp_max}'
""",
        },
    ],
    "solution": """def graph_readout(X, batch, mode='mean'):
    import torch
    num_graphs = batch.max().item() + 1
    D = X.shape[1]
    if mode == 'sum':
        out = torch.zeros(num_graphs, D)
        out.index_add_(0, batch, X)
        return out
    elif mode == 'mean':
        out = torch.zeros(num_graphs, D)
        out.index_add_(0, batch, X)
        count = torch.zeros(num_graphs)
        count.index_add_(0, batch, torch.ones(batch.shape[0]))
        return out / count.unsqueeze(1)
    elif mode == 'max':
        out = torch.full((num_graphs, D), float('-inf'))
        for i in range(num_graphs):
            mask = batch == i
            out[i] = X[mask].max(dim=0).values
        return out
    else:
        raise ValueError(f"Unsupported mode: {mode}")""",
}
