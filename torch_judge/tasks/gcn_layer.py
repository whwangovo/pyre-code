"""GCN Layer (Graph Convolution) task."""

TASK = {
    "title": "GCN Layer (Graph Convolution)",
    "difficulty": "Easy",
    "description_en": "Implement a single Graph Convolutional Network (GCN) layer with symmetric normalization.\n\n**Signature:** `gcn_layer(A, X, W) -> Tensor`\n\n**Parameters:**\n- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops\n- `X` — node feature matrix (N, F_in)\n- `W` — weight matrix (F_in, F_out)\n\n**Returns:** output node features of shape (N, F_out)\n\n**Algorithm:**\n1. Add self-loops: `A_tilde = A + I`\n2. Compute degree vector: `D_vec = A_tilde.sum(1)`\n3. Symmetric normalization: `D_inv_sqrt = diag(D_vec^{-0.5})`\n4. Normalized adjacency: `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`\n5. Return `ReLU(A_norm @ X @ W)`\n\n**Constraints:**\n- Weights `W` are passed in — do not create them inside the function\n- Use `torch.relu` for activation",
    "description_zh": "实现单层图卷积网络 (GCN)，采用对称归一化。\n\n**签名:** `gcn_layer(A, X, W) -> Tensor`\n\n**参数:**\n- `A` — 邻接矩阵 (N, N)，二值、对称、无自环\n- `X` — 节点特征矩阵 (N, F_in)\n- `W` — 权重矩阵 (F_in, F_out)\n\n**返回:** 形状为 (N, F_out) 的输出节点特征\n\n**算法:**\n1. 添加自环：`A_tilde = A + I`\n2. 计算度向量：`D_vec = A_tilde.sum(1)`\n3. 对称归一化：`D_inv_sqrt = diag(D_vec^{-0.5})`\n4. 归一化邻接矩阵：`A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`\n5. 返回 `ReLU(A_norm @ X @ W)`\n\n**约束:**\n- 权重 `W` 由参数传入，不要在函数内部创建\n- 使用 `torch.relu` 作为激活函数",
    "function_name": "gcn_layer",
    "hint": "1. `A_tilde = A + torch.eye(N)`\n2. `D_vec = A_tilde.sum(1)`\n3. `D_inv_sqrt = torch.diag(D_vec.pow(-0.5))`\n4. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`\n5. `return torch.relu(A_norm @ X @ W)`",
    "hint_zh": "1. `A_tilde = A + torch.eye(N)`\n2. `D_vec = A_tilde.sum(1)`\n3. `D_inv_sqrt = torch.diag(D_vec.pow(-0.5))`\n4. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`\n5. `return torch.relu(A_norm @ X @ W)`",
    "tests": [
        {
            "name": "Output shape matches (N, F_out)",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_out = 5, 4, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
out = {fn}(A, X, W)
assert out.shape == (N, F_out), f'Expected ({N},{F_out}), got {out.shape}'
""",
        },
        {
            "name": "Output is non-negative (ReLU applied)",
            "code": """
import torch
torch.manual_seed(0)
N, F_in, F_out = 8, 6, 4
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,7)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
out = {fn}(A, X, W)
assert (out >= 0).all(), f'Output has negative values, ReLU not applied: min={out.min().item()}'
""",
        },
        {
            "name": "Isolated node gets only self-features",
            "code": """
import torch
N, F_in, F_out = 3, 2, 2
A = torch.zeros(N, N)
A[0, 1] = A[1, 0] = 1.0
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
W = torch.eye(F_in)
out = {fn}(A, X, W)
expected_node2 = torch.relu(X[2:3] @ W)
assert torch.allclose(out[2], expected_node2.squeeze(), atol=1e-5), f'Isolated node should only use self-features: got {out[2]}, expected {expected_node2.squeeze()}'
""",
        },
        {
            "name": "Exact numerical value",
            "code": """
import torch
N, F_in, F_out = 3, 2, 2
A = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
X = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
W = torch.tensor([[1.,0.],[0.,1.]])
out = {fn}(A, X, W)
A_tilde = A + torch.eye(N)
D_vec = A_tilde.sum(1)
D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
expected = torch.relu(A_norm @ X @ W)
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff = {(out - expected).abs().max().item():.6f}'
""",
        },
        {
            "name": "Gradient flows through X and W",
            "code": """
import torch
N, F_in, F_out = 4, 3, 2
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in, requires_grad=True)
W = torch.randn(F_in, F_out, requires_grad=True)
out = {fn}(A, X, W)
loss = out.sum()
loss.backward()
assert X.grad is not None and X.grad.abs().sum() > 0, 'No gradient for X'
assert W.grad is not None and W.grad.abs().sum() > 0, 'No gradient for W'
""",
        },
        {
            "name": "Works with different F_in and F_out sizes",
            "code": """
import torch
torch.manual_seed(7)
N, F_in, F_out = 6, 10, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
out = {fn}(A, X, W)
assert out.shape == (N, F_out), f'Expected ({N},{F_out}), got {out.shape}'
assert (out >= 0).all(), 'Output has negative values'
""",
        },
    ],
    "solution": """def gcn_layer(A, X, W):
    import torch
    N = A.shape[0]
    A_tilde = A + torch.eye(N)
    D_vec = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return torch.relu(A_norm @ X @ W)""",
}
