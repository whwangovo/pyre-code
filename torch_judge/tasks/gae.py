"""Graph Autoencoder (GAE) task."""

TASK = {
    "title": "Graph Autoencoder (GAE)",
    "difficulty": "Hard",
    "description_en": "Implement a Graph Autoencoder that learns node embeddings via a two-layer GCN encoder and reconstructs the adjacency matrix with an inner-product decoder.\n\n**Signature:** `gae(A, X) -> Tensor`\n\n**Parameters:**\n- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops\n- `X` — node feature matrix (N, F)\n\n**Returns:** reconstructed adjacency matrix `A_hat` of shape (N, N), with values in [0, 1] (apply sigmoid)\n\n**Architecture:**\n1. Compute degree-normalized adjacency: `A_norm = D^{-1/2} (A + I) D^{-1/2}` where D is the degree matrix of `A + I`\n2. GCN Layer 1: `H = ReLU(A_norm @ X @ W1)` with `W1` shape (F, 32)\n3. GCN Layer 2: `Z = A_norm @ H @ W2` with `Z` shape (N, 16), `W2` shape (32, 16)\n4. Inner-product decoder: `A_hat = sigmoid(Z @ Z^T)`\n\n**Constraints:**\n- Initialize `W1` and `W2` with `torch.manual_seed(0)` then `torch.randn(...) * 0.1` (W1 first, then W2)\n- `W1` shape: (F, 32), `W2` shape: (32, 16)\n- Do not use `nn.Module` or `nn.Linear`\n- Use `torch.relu` and `torch.sigmoid`",
    "description_zh": "实现图自编码器 (GAE)，通过两层 GCN 编码器学习节点嵌入，并用内积解码器重建邻接矩阵。\n\n**签名:** `gae(A, X) -> Tensor`\n\n**参数:**\n- `A` — 邻接矩阵 (N, N)，二值、对称、无自环\n- `X` — 节点特征矩阵 (N, F)\n\n**返回:** 重建的邻接矩阵 `A_hat`，形状 (N, N)，值在 [0, 1] 之间（应用 sigmoid）\n\n**架构:**\n1. 计算度归一化邻接矩阵：`A_norm = D^{-1/2} (A + I) D^{-1/2}`，其中 D 是 `A + I` 的度矩阵\n2. GCN 第一层：`H = ReLU(A_norm @ X @ W1)`，`W1` 形状 (F, 32)\n3. GCN 第二层：`Z = A_norm @ H @ W2`，`Z` 形状 (N, 16)，`W2` 形状 (32, 16)\n4. 内积解码器：`A_hat = sigmoid(Z @ Z^T)`\n\n**约束:**\n- 用 `torch.manual_seed(0)` 初始化 `W1` 和 `W2`，然后依次 `torch.randn(...) * 0.1`（先 W1 后 W2）\n- 不得使用 `nn.Module` 或 `nn.Linear`\n- 使用 `torch.relu` 和 `torch.sigmoid`",
    "function_name": "gae",
    "hint": "1. `A_hat = A + I` (add self-loops)\n2. `D_inv_sqrt = diag(A_hat.sum(1))^{-0.5}`\n3. `A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt`\n4. `W1 = randn(F, 32) * 0.1`, `W2 = randn(32, 16) * 0.1`\n5. `H = relu(A_norm @ X @ W1)`\n6. `Z = A_norm @ H @ W2`\n7. `return sigmoid(Z @ Z.T)`",
    "hint_zh": "1. `A_hat = A + I`（加自环）\n2. `D_inv_sqrt = diag(A_hat.sum(1))^{-0.5}`\n3. `A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt`\n4. `W1 = randn(F, 32) * 0.1`，`W2 = randn(32, 16) * 0.1`\n5. `H = relu(A_norm @ X @ W1)`\n6. `Z = A_norm @ H @ W2`\n7. `return sigmoid(Z @ Z.T)`",
    "tests": [
        {
            "name": "Output shape matches input",
            "code": "\nimport torch\ntorch.manual_seed(42)\nN, F = 6, 8\nA = torch.zeros(N, N)\nedges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]\nfor i,j in edges:\n    A[i,j] = A[j,i] = 1.0\nX = torch.randn(N, F)\nA_hat = {fn}(A, X)\nassert A_hat.shape == (N, N), f'Expected ({N},{N}), got {A_hat.shape}'\n",
        },
        {
            "name": "Output values in [0, 1]",
            "code": "\nimport torch\ntorch.manual_seed(42)\nN, F = 6, 8\nA = torch.zeros(N, N)\nedges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]\nfor i,j in edges:\n    A[i,j] = A[j,i] = 1.0\nX = torch.randn(N, F)\nA_hat = {fn}(A, X)\nassert A_hat.min() >= 0.0, f'Min value {A_hat.min().item()} < 0'\nassert A_hat.max() <= 1.0, f'Max value {A_hat.max().item()} > 1'\n",
        },
        {
            "name": "Output is symmetric",
            "code": "\nimport torch\ntorch.manual_seed(42)\nN, F = 6, 8\nA = torch.zeros(N, N)\nedges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]\nfor i,j in edges:\n    A[i,j] = A[j,i] = 1.0\nX = torch.randn(N, F)\nA_hat = {fn}(A, X)\nassert torch.allclose(A_hat, A_hat.T, atol=1e-5), 'Output should be symmetric since decoder is Z@Z.T'\n",
        },
        {
            "name": "Deterministic with fixed seed",
            "code": "\nimport torch\ntorch.manual_seed(42)\nN, F = 5, 4\nA = torch.zeros(N, N)\nedges = [(0,1),(1,2),(2,3),(3,4)]\nfor i,j in edges:\n    A[i,j] = A[j,i] = 1.0\nX = torch.randn(N, F)\nA_hat1 = {fn}(A, X)\nA_hat2 = {fn}(A, X)\nassert torch.allclose(A_hat1, A_hat2, atol=1e-6), 'Same input should give same output (seed is set inside function)'\n",
        },
        {
            "name": "Exact numerical value",
            "code": """
import torch
N, F = 4, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(0,3)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
torch.manual_seed(7)
X = torch.randn(N, F)
A_hat = {fn}(A, X)
# Compute expected
torch.manual_seed(0)
W1 = torch.randn(F, 32) * 0.1
W2 = torch.randn(32, 16) * 0.1
A_tilde = A + torch.eye(N)
D_vec = A_tilde.sum(1)
D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
H = torch.relu(A_norm @ X @ W1)
Z = A_norm @ H @ W2
expected = torch.sigmoid(Z @ Z.T)
assert torch.allclose(A_hat, expected, atol=1e-5), f'Numerical mismatch: max diff = {(A_hat - expected).abs().max().item():.6f}'
""",
        },
        {
            "name": "Gradient flow through weights",
            "code": "\nimport torch\nN, F = 4, 3\nA = torch.zeros(N, N)\nedges = [(0,1),(1,2),(2,3)]\nfor i,j in edges:\n    A[i,j] = A[j,i] = 1.0\nX = torch.randn(N, F, requires_grad=True)\nA_hat = {fn}(A, X)\nloss = (A_hat - A).pow(2).mean()\nloss.backward()\nassert X.grad is not None, 'No gradient for X'\nassert X.grad.abs().sum() > 0, 'Gradient is all zeros'\n",
        },
    ],
    "solution": '''def gae(A, X):
    import torch
    N = A.shape[0]
    F = X.shape[1]
    torch.manual_seed(0)
    W1 = torch.randn(F, 32) * 0.1
    W2 = torch.randn(32, 16) * 0.1
    A_tilde = A + torch.eye(N)
    D_vec = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    H = torch.relu(A_norm @ X @ W1)
    Z = A_norm @ H @ W2
    return torch.sigmoid(Z @ Z.T)''',
}
