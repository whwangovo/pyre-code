"""ViT Transformer Block task."""

TASK = {
    "title": "ViT Transformer Block",
    "title_zh": "ViT Transformer Block",
    "difficulty": "Hard",
    "description_en": "Implement a Vision Transformer (ViT) block as an nn.Module.\n\nA ViT block uses pre-norm architecture with multi-head self-attention and an MLP, both wrapped in residual connections.\n\n**Signature:** `ViTBlock(d_model, num_heads)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (B, N, d_model), where N = num_patches + 1 (includes CLS token)\n\n**Returns:** output tensor (B, N, d_model)\n\n**Architecture:**\n```\nx = x + MHA(LayerNorm(x))\nx = x + MLP(LayerNorm(x))\n```\n\n**Constraints:**\n- `nn.Linear` and `nn.LayerNorm` are allowed as building blocks\n- MHA must be hand-implemented (no `nn.MultiheadAttention`)\n- MLP: `Linear(d, 4d) -> GELU -> Linear(4d, d)`\n- GELU must be hand-implemented: `x * 0.5 * (1 + erf(x / sqrt(2)))`\n- No `F.*` or `nn.functional.*` calls anywhere",
    "description_zh": "将 Vision Transformer（ViT）块实现为 nn.Module。\n\nViT 块使用 pre-norm 架构，包含多头自注意力和 MLP，两者都有残差连接。\n\n**签名:** `ViTBlock(d_model, num_heads)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (B, N, d_model)，其中 N = 图像块数 + 1（含 CLS token）\n\n**返回:** 输出张量 (B, N, d_model)\n\n**架构:**\n```\nx = x + MHA(LayerNorm(x))\nx = x + MLP(LayerNorm(x))\n```\n\n**约束:**\n- `nn.Linear` 和 `nn.LayerNorm` 可作为构建块使用\n- MHA 必须手动实现（不得使用 `nn.MultiheadAttention`）\n- MLP：`Linear(d, 4d) -> GELU -> Linear(4d, d)`\n- GELU 必须手动实现：`x * 0.5 * (1 + erf(x / sqrt(2)))`\n- 任何地方都不得调用 `F.*` 或 `nn.functional.*`",
    "function_name": "ViTBlock",
    "hint": "MHA: `nn.Linear(d, 3d)` → split Q/K/V → reshape `(B, H, N, d_h)` → scaled dot-product → `proj`\nGELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`\nBlock: `x = x + MHA(norm1(x))` → `x = x + MLP(norm2(x))`",
    "hint_zh": "MHA：`nn.Linear(d, 3d)` → 拆分 Q/K/V → reshape `(B, H, N, d_h)` → 缩放点积 → `proj`\nGELU：`x * 0.5 * (1 + erf(x / sqrt(2)))`\n块结构：`x = x + MHA(norm1(x))` → `x = x + MLP(norm2(x))`",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
block = {fn}(d_model=64, num_heads=4)
assert isinstance(block, nn.Module), 'ViTBlock must inherit from nn.Module'
x = torch.randn(2, 10, 64)
out = block(x)
assert out.shape == (2, 10, 64), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Residual connections active",
            "code": """
import torch
torch.manual_seed(1)
block = {fn}(d_model=32, num_heads=4)
x1 = torch.randn(1, 5, 32)
x2 = x1 + 0.5
out1 = block(x1)
out2 = block(x2)
assert not torch.allclose(out1, out2, atol=1e-4), 'Output unchanged when input changed — residuals may be broken'
""",
        },
        {
            "name": "Pre-norm: has norm1 and norm2 as LayerNorm",
            "code": """
import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'norm1') and isinstance(block.norm1, nn.LayerNorm), 'Need self.norm1 = nn.LayerNorm'
assert hasattr(block, 'norm2') and isinstance(block.norm2, nn.LayerNorm), 'Need self.norm2 = nn.LayerNorm'
""",
        },
        {
            "name": "Gradient flows to all parameters",
            "code": """
import torch
torch.manual_seed(2)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 6, 32, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in block.parameters())
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params received gradients'
""",
        },
        {
            "name": "Works with different num_heads",
            "code": """
import torch
torch.manual_seed(3)
for num_heads in [1, 2, 8]:
    block = {fn}(d_model=64, num_heads=num_heads)
    x = torch.randn(2, 9, 64)
    out = block(x)
    assert out.shape == (2, 9, 64), f'Shape mismatch with num_heads={num_heads}: {out.shape}'
""",
        },
        {
            "name": "Numerical correctness via manual forward pass",
            "code": """
import torch, torch.nn as nn, math
torch.manual_seed(0)
B, S, d_model, num_heads = 1, 4, 16, 2
block = {fn}(d_model=d_model, num_heads=num_heads)
block.eval()
x = torch.randn(B, S, d_model)

# --- manual MHA ---
d_h = d_model // num_heads
normed1 = block.norm1(x)
qkv = block.qkv(normed1).reshape(B, S, 3, num_heads, d_h).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
scale = d_h ** -0.5
attn_w = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
attn_out = (attn_w @ v).transpose(1, 2).reshape(B, S, d_model)
mha_out = block.proj(attn_out)
x_after_attn = x + mha_out

# --- manual MLP with hand-implemented GELU ---
normed2 = block.norm2(x_after_attn)
h = block.fc1(normed2)
gelu_h = h * 0.5 * (1.0 + torch.erf(h / math.sqrt(2.0)))
mlp_out = block.fc2(gelu_h)
expected = x_after_attn + mlp_out

out = block(x)
assert torch.allclose(out, expected, atol=1e-4), f'Max diff: {(out - expected).abs().max().item()}'
""",
        },
    ],
    "solution": """import math as _math

class ViTBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / _math.sqrt(2.0)))

    def _mha(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.d_h ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

    def forward(self, x):
        x = x + self._mha(self.norm1(x))
        x = x + self.fc2(self._gelu(self.fc1(self.norm2(x))))
        return x""",
    "demo": """torch.manual_seed(0)
batch, num_patches, d_model, num_heads = 2, 16, 64, 4

block = ViTBlock(d_model, num_heads)
x = torch.randn(batch, num_patches, d_model)
out = block(x)

print("Input shape: ", x.shape)    # (2, 16, 64)
print("Output shape:", out.shape)  # (2, 16, 64)
assert out.shape == x.shape, "Shape mismatch!"
print("Shape preserved: True")""",

}
