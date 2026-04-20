"""Adaptive LayerNorm Zero (adaLN-Zero) task."""

TASK = {
    "title": "Adaptive LayerNorm Zero (adaLN-Zero)",
    "title_zh": "自适应层归一化 Zero",
    "difficulty": "Medium",
    "description_en": "Implement the adaLN-Zero conditioning mechanism from DiT (Diffusion Transformer, ICCV 2023).\n\nA linear projection of the conditioning embedding (timestep + class label) regresses scale (γ), shift (β), and gate (α) parameters. These modulate the layer-normed input. The gate is zero-initialized so the block starts as an identity function.\n\n**Signature:** `adaln_zero(x, cond, W_ada, b_ada) -> Tensor`\n\n**Parameters:**\n- `x` — input tokens (B, N, D)\n- `cond` — conditioning embedding (B, D)\n- `W_ada` — linear weight (D, 6*D), regresses [γ1, β1, α1, γ2, β2, α2]\n- `b_ada` — bias (6*D,), zero-initialized in practice\n\n**Returns:** modulated tensor (B, N, D) using the first set of params (γ1, β1, α1)\n\n**Steps:**\n1. `params = cond @ W_ada + b_ada` — shape (B, 6*D)\n2. Split into 6 chunks of size D: γ1, β1, α1, γ2, β2, α2\n3. LayerNorm x manually (no learnable affine): subtract mean, divide by std\n4. Modulate: `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`\n5. Return out",
    "description_zh": "实现 DiT（扩散变换器，ICCV 2023）中的 adaLN-Zero 条件机制。\n\n通过对条件嵌入（时间步 + 类别标签）进行线性投影，回归出缩放（γ）、偏移（β）和门控（α）参数，用于调制经过层归一化的输入。门控参数零初始化，使得模块初始时等价于恒等映射。\n\n**签名:** `adaln_zero(x, cond, W_ada, b_ada) -> Tensor`\n\n**参数:**\n- `x` — 输入 token (B, N, D)\n- `cond` — 条件嵌入 (B, D)\n- `W_ada` — 线性权重 (D, 6*D)，回归 [γ1, β1, α1, γ2, β2, α2]\n- `b_ada` — 偏置 (6*D,)，实践中零初始化\n\n**返回:** 使用第一组参数 (γ1, β1, α1) 调制后的张量 (B, N, D)\n\n**步骤:**\n1. `params = cond @ W_ada + b_ada` — 形状 (B, 6*D)\n2. 沿最后一维切分为 6 块，每块大小 D：γ1, β1, α1, γ2, β2, α2\n3. 手动计算 LayerNorm（无可学习仿射参数）：减均值，除以标准差\n4. 调制：`out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`\n5. 返回 out",
    "function_name": "adaln_zero",
    "hint": "1. `params = cond @ W_ada + b_ada`  shape `(B, 6*D)`\n2. `γ1,β1,α1,... = params.chunk(6, dim=-1)`\n3. Manual LayerNorm: `mean/var` over `dim=-1`, `unbiased=False`, `eps=1e-6`\n4. `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`",
    "hint_zh": "1. `params = cond @ W_ada + b_ada`  形状 `(B, 6*D)`\n2. `γ1,β1,α1,... = params.chunk(6, dim=-1)`\n3. 手动 LayerNorm：`dim=-1` 求 `mean/var`，`unbiased=False`，`eps=1e-6`\n4. `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
B, N, D = 2, 16, 64
x = torch.randn(B, N, D)
cond = torch.randn(B, D)
W_ada = torch.randn(D, 6 * D) * 0.02
b_ada = torch.zeros(6 * D)
out = {fn}(x, cond, W_ada, b_ada)
assert out.shape == (B, N, D), f'Expected ({B}, {N}, {D}), got {out.shape}'
"""
        },
        {
            "name": "Zero gate gives zero output",
            "code": """
import torch
torch.manual_seed(1)
B, N, D = 2, 8, 32
x = torch.randn(B, N, D)
cond = torch.randn(B, D)
W_ada = torch.zeros(D, 6 * D)
b_ada = torch.zeros(6 * D)
out = {fn}(x, cond, W_ada, b_ada)
assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), 'With zero W_ada and b_ada, gate=0 so output must be all zeros'
"""
        },
        {
            "name": "Identity modulation (alpha=1, gamma=1, beta=0)",
            "code": """
import torch
torch.manual_seed(2)
B, N, D = 2, 10, 16
x = torch.randn(B, N, D)
cond = torch.randn(B, D)
# Construct b_ada so that alpha1=1, gamma1=1, beta1=0 for all dims
# params = cond @ 0 + b_ada, chunks: gamma1=b[0:D], beta1=b[D:2D], alpha1=b[2D:3D]
b_ada = torch.zeros(6 * D)
b_ada[0:D] = 1.0      # gamma1 = 1
b_ada[D:2*D] = 0.0    # beta1 = 0
b_ada[2*D:3*D] = 1.0  # alpha1 = 1
W_ada = torch.zeros(D, 6 * D)
out = {fn}(x, cond, W_ada, b_ada)
# Compute expected: manual layer norm of x
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm = (x - mean) / (var + 1e-6).sqrt()
assert torch.allclose(out, x_norm, atol=1e-5), 'With alpha=1, gamma=1, beta=0, output should equal layer-normed x'
"""
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
B, N, D = 2, 6, 8
x = torch.randn(B, N, D, requires_grad=True)
cond = torch.randn(B, D, requires_grad=True)
W_ada = torch.nn.Parameter(torch.randn(D, 6 * D))  # leaf tensor
b_ada = torch.zeros(6 * D, requires_grad=True)
out = {fn}(x, cond, W_ada, b_ada)
out.sum().backward()
assert x.grad is not None, 'No gradient for x'
assert cond.grad is not None, 'No gradient for cond'
assert W_ada.grad is not None, 'No gradient for W_ada'
"""
        }
    ],
    "solution": '''def adaln_zero(x, cond, W_ada, b_ada):
    B, N, D = x.shape
    params = cond @ W_ada + b_ada          # (B, 6*D)
    chunks = params.chunk(6, dim=-1)       # 6 x (B, D)
    gamma1, beta1, alpha1 = chunks[0], chunks[1], chunks[2]
    # LayerNorm manually
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / (var + 1e-6).sqrt()
    # Modulate: unsqueeze for broadcast over N tokens
    out = alpha1.unsqueeze(1) * (gamma1.unsqueeze(1) * x_norm + beta1.unsqueeze(1))
    return out''',
    "demo": """torch.manual_seed(0)

B, N, D = 4, 16, 32
C = 16  # conditioning dim

x    = torch.randn(B, N, D)
cond = torch.randn(B, C)

W_ada = torch.zeros(C, 6 * D)
b_ada = torch.zeros(6 * D)
out_zero = adaln_zero(x, cond, W_ada, b_ada)
print(f"Zero W_ada, zero b_ada => max abs output: {out_zero.abs().max().item():.6f}  (expected 0.0)")

W_ada_rand = torch.randn(C, 6 * D) * 0.1
b_ada_rand = torch.randn(6 * D) * 0.1
out_rand = adaln_zero(x, cond, W_ada_rand, b_ada_rand)
print(f"Random W_ada          => output shape: {out_rand.shape}, mean abs: {out_rand.abs().mean().item():.4f}")""",

}
