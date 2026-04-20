"""GraphSAGE Layer task."""

TASK = {
    "title": "GraphSAGE Layer",
    "title_zh": "GraphSAGE 层",
    "difficulty": "Hard",
    "description_en": "Implement a GraphSAGE layer that aggregates neighbor features via mean pooling, concatenates with self features, and applies a linear transformation with L2 normalization.\n\n**Signature:** `graphsage_layer(A, X, W, k=None) -> Tensor`\n\n**Parameters:**\n- `A` — adjacency matrix (N, N), binary\n- `X` — node feature matrix (N, F_in)\n- `W` — weight matrix (2*F_in, F_out)\n- `k` — if not None, sample at most k neighbors per node (use `torch.manual_seed(42)` before sampling for determinism)\n\n**Returns:** (N, F_out) tensor, L2-normalized along the feature dimension\n\n**Algorithm:**\n1. For each node i:\n   - Get neighbor indices where A[i] > 0\n   - If k is not None and num_neighbors > k: set `torch.manual_seed(42)`, use `torch.randperm(num_neighbors)[:k]` to sample\n   - Compute mean of neighbor features (zero vector if no neighbors)\n2. Concatenate [X, neighbor_agg] along feature dim to get (N, 2*F_in)\n3. Apply linear transform: `out = ReLU(concat @ W)`\n4. L2 normalize: `out = out / ||out||_2` per row (clamp norm min to 1e-8)\n\n**Constraints:**\n- Do not use `nn.Module` or `nn.Linear`\n- Use `torch.relu` for activation\n- Sampling must be deterministic with `torch.manual_seed(42)` called immediately before each `randperm`",
    "description_zh": "实现 GraphSAGE 层：通过均值池化聚合邻居特征，与自身特征拼接后进行线性变换，最后做 L2 归一化。\n\n**签名:** `graphsage_layer(A, X, W, k=None) -> Tensor`\n\n**参数:**\n- `A` — 邻接矩阵 (N, N)，二值\n- `X` — 节点特征矩阵 (N, F_in)\n- `W` — 权重矩阵 (2*F_in, F_out)\n- `k` — 若不为 None，每个节点最多采样 k 个邻居（采样前调用 `torch.manual_seed(42)` 保证确定性）\n\n**返回:** (N, F_out) 张量，沿特征维度做 L2 归一化\n\n**算法:**\n1. 对每个节点 i：\n   - 获取 A[i] > 0 的邻居索引\n   - 若 k 不为 None 且邻居数 > k：设置 `torch.manual_seed(42)`，用 `torch.randperm(num_neighbors)[:k]` 采样\n   - 计算邻居特征均值（无邻居则为零向量）\n2. 沿特征维拼接 [X, neighbor_agg] 得到 (N, 2*F_in)\n3. 线性变换：`out = ReLU(concat @ W)`\n4. L2 归一化：`out = out / ||out||_2`（每行归一化，范数下限 clamp 为 1e-8）\n\n**约束:**\n- 不得使用 `nn.Module` 或 `nn.Linear`\n- 使用 `torch.relu` 作为激活函数\n- 采样必须确定性：每次 `randperm` 前立即调用 `torch.manual_seed(42)`",
    "function_name": "graphsage_layer",
    "hint": "1. Loop over nodes, gather neighbors from A[i].nonzero()\n2. If k sampling needed: `torch.manual_seed(42)`, `perm = torch.randperm(n_neighbors)[:k]`\n3. Mean-pool neighbor features (or zeros if isolated)\n4. `concat = torch.cat([X, agg], dim=1)` → (N, 2*F_in)\n5. `out = relu(concat @ W)`\n6. `out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)`",
    "hint_zh": "1. 遍历节点，从 A[i].nonzero() 获取邻居\n2. 若需采样：`torch.manual_seed(42)`，`perm = torch.randperm(n_neighbors)[:k]`\n3. 均值池化邻居特征（孤立节点用零向量）\n4. `concat = torch.cat([X, agg], dim=1)` → (N, 2*F_in)\n5. `out = relu(concat @ W)`\n6. `out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)`",
    "tests": [
        {
            "name": "Output shape check",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_out = 5, 4, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out = {fn}(A, X, W)
assert out.shape == (N, F_out), f'Expected shape ({N},{F_out}), got {out.shape}'
""",
        },
        {
            "name": "Output is L2-normalized",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_out = 6, 4, 8
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out = {fn}(A, X, W)
norms = out.norm(dim=-1)
# Rows with non-zero output should have unit norm
nonzero_mask = norms > 1e-7
assert torch.allclose(norms[nonzero_mask], torch.ones_like(norms[nonzero_mask]), atol=1e-5), f'Output rows not L2-normalized: norms = {norms}'
""",
        },
        {
            "name": "k=None uses all neighbors",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_out = 4, 3, 2
A = torch.zeros(N, N)
# Node 0 connected to all others
A[0,1] = A[1,0] = 1.0
A[0,2] = A[2,0] = 1.0
A[0,3] = A[3,0] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out_none = {fn}(A, X, W, k=None)
out_large_k = {fn}(A, X, W, k=100)
assert torch.allclose(out_none, out_large_k, atol=1e-6), 'k=None and k>num_neighbors should give same result'
""",
        },
        {
            "name": "k sampling is deterministic",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_out = 8, 4, 3
A = torch.zeros(N, N)
# Dense graph
for i in range(N):
    for j in range(i+1, N):
        A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out1 = {fn}(A, X, W, k=2)
out2 = {fn}(A, X, W, k=2)
assert torch.allclose(out1, out2, atol=1e-6), 'k-sampling should be deterministic with manual_seed(42)'
""",
        },
        {
            "name": "Exact numerical value",
            "code": """
import torch
N, F_in, F_out = 3, 2, 2
A = torch.tensor([[0.,1.,1.],[1.,0.,0.],[1.,0.,0.]])
X = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
W = torch.tensor([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
out = {fn}(A, X, W)
# Manual computation
# Node 0 neighbors: [1,2], mean = ([0,1]+[1,1])/2 = [0.5, 1.0]
# Node 1 neighbors: [0], mean = [1, 0]
# Node 2 neighbors: [0], mean = [1, 0]
agg = torch.tensor([[0.5, 1.0],[1.0, 0.0],[1.0, 0.0]])
concat = torch.cat([X, agg], dim=1)  # (3, 4)
raw = torch.relu(concat @ W)  # (3, 2)
expected = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff = {(out - expected).abs().max().item():.6f}'
""",
        },
        {
            "name": "Gradient flow through X and W",
            "code": """
import torch
torch.manual_seed(1)
N, F_in, F_out = 4, 3, 2
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in, requires_grad=True)
W = torch.randn(2*F_in, F_out, requires_grad=True)
out = {fn}(A, X, W)
loss = out.sum()
loss.backward()
assert X.grad is not None, 'No gradient for X'
assert X.grad.abs().sum() > 0, 'X gradient is all zeros'
assert W.grad is not None, 'No gradient for W'
assert W.grad.abs().sum() > 0, 'W gradient is all zeros'
""",
        },
        {
            "name": "k < num_neighbors changes aggregation",
            "code": """
import torch
N, F_in, F_out = 6, 4, 3
A = torch.zeros(N, N)
# Node 0 connected to all others (5 neighbors)
for j in range(1, N):
    A[0,j] = A[j,0] = 1.0
# Use distinct features so any subset gives a different mean
X = torch.tensor([[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,2.,0.,0.],[0.,0.,3.,0.],[0.,0.,0.,4.],[5.,5.,5.,5.]])
W = torch.eye(2*F_in, F_out)[:2*F_in, :F_out]
W = torch.randn(2*F_in, F_out)
torch.manual_seed(99)
W = torch.randn(2*F_in, F_out)
out_all = {fn}(A, X, W, k=None)
out_k2 = {fn}(A, X, W, k=2)
# Node 0 has 5 neighbors with distinct features; k=2 samples a subset, giving different aggregation
assert not torch.allclose(out_all[0], out_k2[0], atol=1e-5), 'k=2 with 5 distinct neighbors should differ from using all neighbors'
""",
        },
        {
            "name": "Isolated node gets zero neighbor aggregation",
            "code": """
import torch
N, F_in, F_out = 3, 2, 2
A = torch.zeros(N, N)
A[0,1] = A[1,0] = 1.0
# Node 2 is isolated
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[2.0, 3.0]])
W = torch.tensor([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
out = {fn}(A, X, W)
# Node 2: neighbors=[], agg=zeros(2), concat=[2,3,0,0]
concat_2 = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
raw_2 = torch.relu(concat_2 @ W)
expected_2 = raw_2 / raw_2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
assert torch.allclose(out[2], expected_2.squeeze(), atol=1e-5), f'Isolated node mismatch: {out[2]} vs {expected_2.squeeze()}'
""",
        },
    ],
    "solution": '''def graphsage_layer(A, X, W, k=None):
    import torch
    N = A.shape[0]
    F_in = X.shape[1]
    agg_list = []
    for i in range(N):
        neighbors = (A[i] > 0).nonzero(as_tuple=True)[0]
        if len(neighbors) == 0:
            agg_list.append(torch.zeros(F_in))
        else:
            if k is not None and len(neighbors) > k:
                torch.manual_seed(42)
                perm = torch.randperm(len(neighbors))[:k]
                neighbors = neighbors[perm]
            agg_list.append(X[neighbors].mean(dim=0))
    agg = torch.stack(agg_list)
    concat = torch.cat([X, agg], dim=1)
    out = torch.relu(concat @ W)
    out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return out''',
}
