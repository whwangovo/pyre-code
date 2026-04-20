"""Link Prediction task."""

TASK = {
    "title": "Link Prediction",
    "title_zh": "链接预测",
    "difficulty": "Hard",
    "description_en": "Implement a link prediction model using a two-layer GCN encoder that predicts edge existence probabilities.\n\n**Signature:** `link_prediction(A, X, edges) -> Tensor`\n\n**Parameters:**\n- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops\n- `X` — node feature matrix (N, F)\n- `edges` — (M, 2) long tensor of node pairs to predict\n\n**Returns:** (M,) tensor of probabilities in [0, 1], where each value is the predicted likelihood of an edge existing between the corresponding node pair\n\n**Algorithm:**\n1. Initialize weights with `torch.manual_seed(0)`: `W1 = randn(F, 16) * 0.1`, then `W2 = randn(16, 8) * 0.1`\n2. Compute degree-normalized adjacency: `A_norm = D^{-1/2} (A + I) D^{-1/2}`\n3. GCN Layer 1: `H = ReLU(A_norm @ X @ W1)`\n4. GCN Layer 2: `Z = A_norm @ H @ W2` (no activation)\n5. For each edge (i, j): `score = sigmoid(Z[i] · Z[j])`\n6. Return all scores as (M,) tensor\n\n**Constraints:**\n- Do not use `nn.Module` or `nn.Linear`\n- Use `torch.relu` and `torch.sigmoid`",
    "description_zh": "实现基于两层 GCN 编码器的链接预测模型，预测节点对之间存在边的概率。\n\n**签名:** `link_prediction(A, X, edges) -> Tensor`\n\n**参数:**\n- `A` — 邻接矩阵 (N, N)，二值、对称、无自环\n- `X` — 节点特征矩阵 (N, F)\n- `edges` — (M, 2) 长整型张量，表示待预测的节点对\n\n**返回:** (M,) 概率张量，值在 [0, 1] 之间，每个值表示对应节点对之间存在边的预测概率\n\n**算法:**\n1. 用 `torch.manual_seed(0)` 初始化权重：`W1 = randn(F, 16) * 0.1`，然后 `W2 = randn(16, 8) * 0.1`\n2. 计算度归一化邻接矩阵：`A_norm = D^{-1/2} (A + I) D^{-1/2}`\n3. GCN 第一层：`H = ReLU(A_norm @ X @ W1)`\n4. GCN 第二层：`Z = A_norm @ H @ W2`（无激活函数）\n5. 对每条边 (i, j)：`score = sigmoid(Z[i] · Z[j])`\n6. 返回所有分数组成的 (M,) 张量\n\n**约束:**\n- 不得使用 `nn.Module` 或 `nn.Linear`\n- 使用 `torch.relu` 和 `torch.sigmoid`",
    "function_name": "link_prediction",
    "hint": "1. `A_tilde = A + I`\n2. `D_inv_sqrt = diag(A_tilde.sum(1))^{-0.5}`\n3. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`\n4. Two-layer GCN: H = relu(A_norm @ X @ W1), Z = A_norm @ H @ W2\n5. Score each edge: `sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))`",
    "hint_zh": "1. `A_tilde = A + I`（加自环）\n2. `D_inv_sqrt = diag(A_tilde.sum(1))^{-0.5}`\n3. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`\n4. 两层 GCN：H = relu(A_norm @ X @ W1)，Z = A_norm @ H @ W2\n5. 对每条边打分：`sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))`",
    "tests": [
        {
            "name": "Output shape is (M,)",
            "code": """
import torch
torch.manual_seed(42)
N, F = 6, 8
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
edges = torch.tensor([[0,1],[2,3],[0,4],[1,5]], dtype=torch.long)
out = {fn}(A, X, edges)
assert out.shape == (4,), f'Expected shape (4,), got {out.shape}'
""",
        },
        {
            "name": "Values in [0, 1]",
            "code": """
import torch
torch.manual_seed(42)
N, F = 6, 8
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
edges = torch.tensor([[0,1],[2,3],[0,4],[1,5],[3,5],[0,2]], dtype=torch.long)
out = {fn}(A, X, edges)
assert out.min() >= 0.0, f'Min value {out.min().item()} < 0'
assert out.max() <= 1.0, f'Max value {out.max().item()} > 1'
""",
        },
        {
            "name": "Deterministic with same input",
            "code": """
import torch
torch.manual_seed(42)
N, F = 5, 4
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
edges = torch.tensor([[0,2],[1,3],[0,4]], dtype=torch.long)
out1 = {fn}(A, X, edges)
out2 = {fn}(A, X, edges)
assert torch.allclose(out1, out2, atol=1e-6), 'Same input should give same output (seed is set inside function)'
""",
        },
        {
            "name": "Exact numerical value",
            "code": """
import torch
N, F = 4, 3
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(0,3)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
torch.manual_seed(7)
X = torch.randn(N, F)
edges = torch.tensor([[0,1],[1,2],[0,2]], dtype=torch.long)
out = {fn}(A, X, edges)
# Compute expected
torch.manual_seed(0)
W1 = torch.randn(F, 16) * 0.1
W2 = torch.randn(16, 8) * 0.1
A_tilde = A + torch.eye(N)
D_vec = A_tilde.sum(1)
D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
H = torch.relu(A_norm @ X @ W1)
Z = A_norm @ H @ W2
scores = torch.sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))
assert torch.allclose(out, scores, atol=1e-5), f'Numerical mismatch: max diff = {(out - scores).abs().max().item():.6f}'
""",
        },
        {
            "name": "Gradient flow through X",
            "code": """
import torch
N, F = 4, 3
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F, requires_grad=True)
edges = torch.tensor([[0,1],[1,2],[0,3]], dtype=torch.long)
out = {fn}(A, X, edges)
loss = out.sum()
loss.backward()
assert X.grad is not None, 'No gradient for X'
assert X.grad.abs().sum() > 0, 'Gradient is all zeros'
""",
        },
        {
            "name": "Symmetric node pairs get same score",
            "code": """
import torch
N, F = 6, 4
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
torch.manual_seed(42)
X = torch.randn(N, F)
# (i,j) and (j,i) should produce the same score since decoder is dot product
edges_fwd = torch.tensor([[0,1],[2,3],[1,4]], dtype=torch.long)
edges_rev = torch.tensor([[1,0],[3,2],[4,1]], dtype=torch.long)
scores_fwd = {fn}(A, X, edges_fwd)
scores_rev = {fn}(A, X, edges_rev)
assert torch.allclose(scores_fwd, scores_rev, atol=1e-5), f'score(i,j) should equal score(j,i): max diff = {(scores_fwd - scores_rev).abs().max().item():.6f}'
""",
        },
    ],
    "solution": '''def link_prediction(A, X, edges):
    import torch
    N = A.shape[0]
    F = X.shape[1]
    torch.manual_seed(0)
    W1 = torch.randn(F, 16) * 0.1
    W2 = torch.randn(16, 8) * 0.1
    A_tilde = A + torch.eye(N)
    D_vec = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    H = torch.relu(A_norm @ X @ W1)
    Z = A_norm @ H @ W2
    scores = torch.sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))
    return scores''',
}
