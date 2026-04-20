"""MPNN Layer (Message Passing Neural Network) task."""

TASK = {
    "title": "MPNN Layer (Message Passing)",
    "difficulty": "Medium",
    "description_en": "Implement a Message Passing Neural Network (MPNN) layer.\n\nMPNN computes messages along edges using node and edge features, aggregates them per node, then updates node representations.\n\n**Signature:** `mpnn_layer(A, X, E, W_msg, W_upd) -> Tensor`\n\n**Parameters:**\n- `A` — adjacency matrix (N, N), binary\n- `X` — node feature matrix (N, F)\n- `E` — edge feature tensor (N, N, D_e)\n- `W_msg` — message weight matrix (2*F + D_e, F_msg)\n- `W_upd` — update weight matrix (F + F_msg, F_out)\n\n**Returns:** updated node features (N, F_out)\n\n**Algorithm:**\n1. For each edge (i,j) where A[i,j]=1, compute message: `m_ij = ReLU(concat(X[i], X[j], E[i,j]) @ W_msg)`\n2. For each node i, aggregate: `agg_i = sum of m_ij over neighbors j`\n3. For each node i, update: `out_i = ReLU(concat(X[i], agg_i) @ W_upd)`\n\n**Hint:** Build an (N, N, 2F+D_e) tensor by broadcasting X, then matmul with W_msg, mask by A, and sum over dim=1.",
    "description_zh": "实现消息传递神经网络（MPNN）层。\n\nMPNN 利用节点和边特征沿边计算消息，按节点聚合后更新节点表示。\n\n**签名:** `mpnn_layer(A, X, E, W_msg, W_upd) -> Tensor`\n\n**参数:**\n- `A` — 邻接矩阵 (N, N)，二值\n- `X` — 节点特征矩阵 (N, F)\n- `E` — 边特征张量 (N, N, D_e)\n- `W_msg` — 消息权重矩阵 (2*F + D_e, F_msg)\n- `W_upd` — 更新权重矩阵 (F + F_msg, F_out)\n\n**返回:** 更新后的节点特征 (N, F_out)\n\n**算法:**\n1. 对每条边 (i,j)（A[i,j]=1），计算消息：`m_ij = ReLU(concat(X[i], X[j], E[i,j]) @ W_msg)`\n2. 对每个节点 i，聚合：`agg_i = 对邻居 j 的 m_ij 求和`\n3. 对每个节点 i，更新：`out_i = ReLU(concat(X[i], agg_i) @ W_upd)`\n\n**提示:** 利用广播构建 (N, N, 2F+D_e) 张量，与 W_msg 矩阵乘，用 A 掩码后沿 dim=1 求和。",
    "function_name": "mpnn_layer",
    "hint": "Build (N, N, 2F+D_e) by broadcasting `X[i]` and `X[j]` with `E[i,j]`, matmul with W_msg, mask by A, sum over dim=1.",
    "hint_zh": "利用广播构建 (N, N, 2F+D_e) 张量，拼接 `X[i]`、`X[j]` 和 `E[i,j]`，与 W_msg 矩阵乘，用 A 掩码后沿 dim=1 求和。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(42)
N, F, D_e, F_msg, F_out = 5, 4, 3, 8, 6
A = (torch.rand(N, N) > 0.5).float()
X = torch.randn(N, F)
E = torch.randn(N, N, D_e)
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape} vs {(N, F_out)}'
""",
        },
        {
            "name": "Exact numerical value",
            "code": """
import torch
torch.manual_seed(100)
N, F, D_e, F_msg, F_out = 3, 2, 1, 4, 2
A = torch.tensor([[1,1,0],[1,1,1],[0,1,1]], dtype=torch.float)
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
E = torch.zeros(N, N, D_e)
E[0,1,0] = 0.5; E[1,0,0] = -0.5; E[1,2,0] = 1.0; E[2,1,0] = -1.0
E[0,0,0] = 0.1; E[1,1,0] = 0.2; E[2,2,0] = 0.3
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
# Reference
Xi = X.unsqueeze(1).expand(N, N, F)
Xj = X.unsqueeze(0).expand(N, N, F)
inp = torch.cat([Xi, Xj, E], dim=-1)
msgs = torch.relu(inp @ W_msg)
msgs = msgs * A.unsqueeze(-1)
agg = msgs.sum(dim=1)
upd_inp = torch.cat([X, agg], dim=-1)
ref = torch.relu(upd_inp @ W_upd)
assert torch.allclose(out, ref, atol=1e-5), f'Value mismatch:\\n{out}\\nvs\\n{ref}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(42)
N, F, D_e, F_msg, F_out = 4, 3, 2, 6, 3
A = torch.ones(N, N)
X = torch.randn(N, F, requires_grad=True)
E = torch.randn(N, N, D_e, requires_grad=True)
W_msg = torch.randn(2 * F + D_e, F_msg, requires_grad=True)
W_upd = torch.randn(F + F_msg, F_out, requires_grad=True)
out = {fn}(A, X, E, W_msg, W_upd)
out.sum().backward()
assert X.grad is not None, 'X.grad is None'
assert E.grad is not None, 'E.grad is None'
assert W_msg.grad is not None, 'W_msg.grad is None'
assert W_upd.grad is not None, 'W_upd.grad is None'
""",
        },
        {
            "name": "Single edge graph",
            "code": """
import torch
torch.manual_seed(7)
N, F, D_e, F_msg, F_out = 3, 2, 1, 4, 2
A = torch.zeros(N, N)
A[0, 1] = 1.0
X = torch.randn(N, F)
E = torch.randn(N, N, D_e)
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
# Node 0 has one outgoing edge to node 1
msg_01 = torch.relu(torch.cat([X[0], X[1], E[0,1]]) @ W_msg)
agg_0 = msg_01
ref_0 = torch.relu(torch.cat([X[0], agg_0]) @ W_upd)
assert torch.allclose(out[0], ref_0, atol=1e-5), f'Node 0 mismatch: {out[0]} vs {ref_0}'
# Node 2 has no neighbors
agg_2 = torch.zeros(F_msg)
ref_2 = torch.relu(torch.cat([X[2], agg_2]) @ W_upd)
assert torch.allclose(out[2], ref_2, atol=1e-5), f'Node 2 (no neighbors) mismatch: {out[2]} vs {ref_2}'
""",
        },
        {
            "name": "Symmetric graph behavior",
            "code": """
import torch
torch.manual_seed(88)
N, F, D_e, F_msg, F_out = 4, 3, 2, 5, 3
A = torch.ones(N, N)
X = torch.randn(N, F)
E_sym = torch.randn(N, N, D_e)
E_sym = (E_sym + E_sym.transpose(0, 1)) / 2
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E_sym, W_msg, W_upd)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape}'
assert not torch.isnan(out).any(), 'Output contains NaN'
assert not torch.isinf(out).any(), 'Output contains Inf'
""",
        },
        {
            "name": "No edges (empty graph)",
            "code": """
import torch
torch.manual_seed(11)
N, F, D_e, F_msg, F_out = 3, 2, 1, 4, 2
A = torch.zeros(N, N)
X = torch.randn(N, F)
E = torch.randn(N, N, D_e)
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
agg = torch.zeros(N, F_msg)
ref = torch.relu(torch.cat([X, agg], dim=-1) @ W_upd)
assert torch.allclose(out, ref, atol=1e-5), f'Empty graph mismatch:\\n{out}\\nvs\\n{ref}'
""",
        },
    ],
    "solution": '''def mpnn_layer(A, X, E, W_msg, W_upd):
    N, F = X.shape
    Xi = X.unsqueeze(1).expand(N, N, F)
    Xj = X.unsqueeze(0).expand(N, N, F)
    inp = torch.cat([Xi, Xj, E], dim=-1)
    msgs = torch.relu(inp @ W_msg)
    msgs = msgs * A.unsqueeze(-1)
    agg = msgs.sum(dim=1)
    return torch.relu(torch.cat([X, agg], dim=-1) @ W_upd)''',
}
