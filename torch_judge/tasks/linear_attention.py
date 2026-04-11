"""Linear Self-Attention task."""

TASK = {
    "title": "Linear Self-Attention",
    "difficulty": "Hard",
    "description_en": "Implement linear self-attention with kernel feature maps.\n\nLinear attention replaces softmax with a feature map phi, enabling O(S*D^2) complexity instead of O(S^2*D) by reordering the computation.\n\n**Signature:** `linear_attention(Q, K, V) -> Tensor`\n\n**Parameters:**\n- `Q` — query tensor (B, S, D_k)\n- `K` — key tensor (B, S, D_k)\n- `V` — value tensor (B, S, D_v)\n\n**Returns:** attention output (B, S, D_v)\n\n**Constraints:**\n- Feature map: `phi(x) = elu(x) + 1`\n- Compute `phi(Q) @ (phi(K)^T @ V)` not `softmax(Q @ K^T) @ V`\n- Normalize by `phi(Q) @ sum(phi(K))` (add `eps=1e-6` for numerical stability)",
    "description_zh": "实现基于核特征映射的线性自注意力。\n\n线性注意力用特征映射 phi 替代 softmax，通过重排计算顺序将复杂度从 O(S^2*D) 降至 O(S*D^2)。\n\n**签名:** `linear_attention(Q, K, V) -> Tensor`\n\n**参数:**\n- `Q` — 查询张量 (B, S, D_k)\n- `K` — 键张量 (B, S, D_k)\n- `V` — 值张量 (B, S, D_v)\n\n**返回:** 注意力输出 (B, S, D_v)\n\n**约束:**\n- 特征映射：`phi(x) = elu(x) + 1`\n- 计算 `phi(Q) @ (phi(K)^T @ V)` 而非 `softmax(Q @ K^T) @ V`\n- 通过 `phi(Q) @ sum(phi(K))` 归一化（加 `eps=1e-6` 保证数值稳定）",
    "function_name": "linear_attention",
    "hint": "`phi(x) = elu(x) + 1`. Let `KV = phi(K).transpose(-2, -1) @ V` and `Z = phi(K).sum(dim=-2, keepdim=True)`. Output = `(phi(Q) @ KV) / (phi(Q) @ Z.transpose(-2, -1) + 1e-6)`.",
    "hint_zh": "`phi(x) = elu(x) + 1`。令 `KV = phi(K).transpose(-2, -1) @ V`，`Z = phi(K).sum(dim=-2, keepdim=True)`。输出 = `(phi(Q) @ KV) / (phi(Q) @ Z.transpose(-2, -1) + 1e-6)`。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "No NaN or Inf",
            "code": """
import torch
torch.manual_seed(0)
out = {fn}(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8))
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
Q = torch.randn(1, 4, 8, requires_grad=True)
K = torch.randn(1, 4, 8, requires_grad=True)
V = torch.randn(1, 4, 8, requires_grad=True)
{fn}(Q, K, V).sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
        },
        {
            "name": "Runs fast on long sequences (linear complexity)",
            "code": """
import torch, time
torch.manual_seed(0)
Q = torch.randn(1, 2048, 64)
K = torch.randn(1, 2048, 64)
V = torch.randn(1, 2048, 64)
t0 = time.perf_counter()
for _ in range(10):
    {fn}(Q, K, V)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s — should be O(S*D^2) not O(S^2*D)'
""",
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch
torch.manual_seed(7)
B, S, D = 1, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
# Reference: phi(x) = elu(x) + 1, elu(x) = x if x>0 else exp(x)-1
def elu_plus_1(x):
    return torch.where(x > 0, x + 1, x.exp())
phi_Q = elu_plus_1(Q)
phi_K = elu_plus_1(K)
KV = phi_K.transpose(-2, -1) @ V          # (B, D, D_v)
Z = phi_K.sum(dim=1, keepdim=True)        # (B, 1, D)
num = phi_Q @ KV                           # (B, S, D_v)
den = phi_Q @ Z.transpose(-2, -1)         # (B, S, 1)
expected = num / (den + 1e-6)
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff {(out - expected).abs().max()}'
""",
        },
    ],
    "solution": '''def linear_attention(Q, K, V):
    Q_prime = torch.where(Q > 0, Q, torch.exp(Q) - 1) + 1
    K_prime = torch.where(K > 0, K, torch.exp(K) - 1) + 1
    KV = torch.bmm(K_prime.transpose(1, 2), V)       # (B, D_k, D_v)
    Z = K_prime.sum(dim=1, keepdim=True)              # (B, 1, D_k)
    num = torch.bmm(Q_prime, KV)                      # (B, S, D_v)
    den = torch.bmm(Q_prime, Z.transpose(1, 2))       # (B, S, 1)
    return num / (den + 1e-6)''',
}
