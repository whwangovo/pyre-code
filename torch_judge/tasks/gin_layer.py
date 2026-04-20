"""GIN Layer (Graph Isomorphism Network) task."""

TASK = {
    "title": "GIN Layer (Graph Isomorphism Network)",
    "title_zh": "GIN 层（图同构网络）",
    "difficulty": "Medium",
    "description_en": "Implement a Graph Isomorphism Network (GIN) layer.\n\nGIN updates node features by aggregating neighbor features, scaling the node's own features by (1 + eps), and passing through a 2-layer MLP.\n\n**Signature:** `gin_layer(A, X, eps, W1, b1, W2, b2) -> Tensor`\n\n**Parameters:**\n- `A` — adjacency matrix (N, N), binary\n- `X` — node feature matrix (N, F_in)\n- `eps` — learnable scalar tensor\n- `W1` — first linear weight (F_in, F_hidden)\n- `b1` — first linear bias (F_hidden,)\n- `W2` — second linear weight (F_hidden, F_out)\n- `b2` — second linear bias (F_out,)\n\n**Returns:** updated node features (N, F_out)\n\n**Algorithm:**\n1. Aggregate neighbor features: `agg = A @ X`\n2. Combine: `h = (1 + eps) * X + agg`\n3. MLP layer 1: `h = ReLU(h @ W1 + b1)`\n4. MLP layer 2: `h = h @ W2 + b2`\n5. Return h",
    "description_zh": "实现图同构网络（GIN）层。\n\nGIN 通过聚合邻居特征、用 (1 + eps) 缩放自身特征，再经过两层 MLP 来更新节点表示。\n\n**签名:** `gin_layer(A, X, eps, W1, b1, W2, b2) -> Tensor`\n\n**参数:**\n- `A` — 邻接矩阵 (N, N)，二值\n- `X` — 节点特征矩阵 (N, F_in)\n- `eps` — 可学习标量张量\n- `W1` — 第一层线性权重 (F_in, F_hidden)\n- `b1` — 第一层线性偏置 (F_hidden,)\n- `W2` — 第二层线性权重 (F_hidden, F_out)\n- `b2` — 第二层线性偏置 (F_out,)\n\n**返回:** 更新后的节点特征 (N, F_out)\n\n**算法:**\n1. 聚合邻居特征：`agg = A @ X`\n2. 组合：`h = (1 + eps) * X + agg`\n3. MLP 第一层：`h = ReLU(h @ W1 + b1)`\n4. MLP 第二层：`h = h @ W2 + b2`\n5. 返回 h",
    "function_name": "gin_layer",
    "hint": "Aggregate with `A @ X`, combine as `(1 + eps) * X + agg`, then pass through a 2-layer MLP with ReLU.",
    "hint_zh": "用 `A @ X` 聚合邻居，组合为 `(1 + eps) * X + agg`，然后通过带 ReLU 的两层 MLP。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_hid, F_out = 5, 8, 16, 4
A = torch.ones(N, N)
X = torch.randn(N, F_in)
eps = torch.tensor(0.0)
W1 = torch.randn(F_in, F_hid)
b1 = torch.randn(F_hid)
W2 = torch.randn(F_hid, F_out)
b2 = torch.randn(F_out)
out = {fn}(A, X, eps, W1, b1, W2, b2)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape} vs {(N, F_out)}'
""",
        },
        {
            "name": "eps=0 behavior",
            "code": """
import torch
torch.manual_seed(10)
N, F_in, F_hid, F_out = 3, 4, 8, 2
A = torch.eye(N)
X = torch.randn(N, F_in)
eps = torch.tensor(0.0)
W1 = torch.randn(F_in, F_hid)
b1 = torch.zeros(F_hid)
W2 = torch.randn(F_hid, F_out)
b2 = torch.zeros(F_out)
out = {fn}(A, X, eps, W1, b1, W2, b2)
h = X + A @ X
h = torch.relu(h @ W1 + b1)
ref = h @ W2 + b2
assert torch.allclose(out, ref, atol=1e-5), f'eps=0 with identity A should give (1+0)*X + I@X = 2X through MLP'
""",
        },
        {
            "name": "Exact numerical value",
            "code": """
import torch
torch.manual_seed(77)
A = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float)
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
eps = torch.tensor(0.5)
W1 = torch.tensor([[1.0, -1.0, 0.5],[0.5, 1.0, -0.5]])
b1 = torch.tensor([0.0, 0.0, 0.0])
W2 = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
b2 = torch.tensor([0.0, 0.0])
out = {fn}(A, X, eps, W1, b1, W2, b2)
agg = A @ X
h = (1 + eps) * X + agg
h = torch.relu(h @ W1 + b1)
ref = h @ W2 + b2
assert torch.allclose(out, ref, atol=1e-5), f'Value mismatch:\\n{out}\\nvs\\n{ref}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_hid, F_out = 4, 6, 8, 3
A = torch.ones(N, N)
X = torch.randn(N, F_in, requires_grad=True)
eps = torch.tensor(0.1, requires_grad=True)
W1 = torch.randn(F_in, F_hid, requires_grad=True)
b1 = torch.randn(F_hid, requires_grad=True)
W2 = torch.randn(F_hid, F_out, requires_grad=True)
b2 = torch.randn(F_out, requires_grad=True)
out = {fn}(A, X, eps, W1, b1, W2, b2)
out.sum().backward()
assert X.grad is not None, 'X.grad is None'
assert eps.grad is not None, 'eps.grad is None'
assert W1.grad is not None, 'W1.grad is None'
assert W2.grad is not None, 'W2.grad is None'
""",
        },
        {
            "name": "Isolated node (no neighbors)",
            "code": """
import torch
torch.manual_seed(55)
A = torch.tensor([[0,0,0],[0,0,1],[0,1,0]], dtype=torch.float)
X = torch.tensor([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0]])
eps = torch.tensor(0.0)
W1 = torch.ones(2, 2)
b1 = torch.zeros(2)
W2 = torch.ones(2, 1)
b2 = torch.zeros(1)
out = {fn}(A, X, eps, W1, b1, W2, b2)
h0 = (1 + 0.0) * X[0] + torch.zeros(2)
h0 = torch.relu(h0 @ W1[:, :] + b1)
ref0 = h0 @ W2 + b2
assert torch.allclose(out[0], ref0, atol=1e-5), f'Isolated node output wrong: {out[0]} vs {ref0}'
""",
        },
        {
            "name": "Different dimensions",
            "code": """
import torch
torch.manual_seed(33)
for F_in, F_hid, F_out in [(2, 4, 1), (8, 16, 8), (3, 6, 2)]:
    N = 4
    A = (torch.rand(N, N) > 0.3).float()
    X = torch.randn(N, F_in)
    eps = torch.tensor(0.1)
    W1 = torch.randn(F_in, F_hid)
    b1 = torch.randn(F_hid)
    W2 = torch.randn(F_hid, F_out)
    b2 = torch.randn(F_out)
    out = {fn}(A, X, eps, W1, b1, W2, b2)
    assert out.shape == (N, F_out), f'Shape mismatch for dims ({F_in},{F_hid},{F_out}): {out.shape}'
""",
        },
    ],
    "solution": '''def gin_layer(A, X, eps, W1, b1, W2, b2):
    agg = A @ X
    h = (1 + eps) * X + agg
    h = torch.relu(h @ W1 + b1)
    return h @ W2 + b2''',
}
