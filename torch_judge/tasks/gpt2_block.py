"""GPT-2 Transformer Block task."""

TASK = {
    "title": "GPT-2 Transformer Block",
    "title_zh": "GPT-2 Transformer Block",
    "difficulty": "Hard",
    "description_en": "Implement a GPT-2 transformer block as an nn.Module.\n\nA GPT-2 block uses pre-norm architecture: LayerNorm before causal self-attention and MLP, with residual connections around both.\n\n**Signature:** `GPT2Block(d_model, num_heads)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (B, S, d_model)\n\n**Returns:** output tensor (B, S, d_model)\n\n**Constraints:**\n- Pre-norm: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`\n- MLP: Linear(d, 4d) -> GELU -> Linear(4d, d)\n- Attention must be causal (future tokens cannot affect past)",
    "description_zh": "实现 GPT-2 Transformer 块（nn.Module）。\n\nGPT-2 块使用 pre-norm 架构：在因果自注意力和 MLP 之前进行 LayerNorm，两者都有残差连接。\n\n**签名:** `GPT2Block(d_model, num_heads)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (B, S, d_model)\n\n**返回:** 输出张量 (B, S, d_model)\n\n**约束:**\n- Pre-norm：`x = x + attn(ln1(x))`，`x = x + mlp(ln2(x))`\n- MLP：Linear(d, 4d) -> GELU -> Linear(4d, d)\n- 注意力必须是因果的（未来 token 不能影响过去）",
    "function_name": "GPT2Block",
    "hint": "Pre-norm residual: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`. MLP: `Linear(d,4d) → GELU → Linear(4d,d)`. Attention must be causal (mask future with `-inf`).",
    "hint_zh": "Pre-norm 残差：`x = x + attn(ln1(x))`，`x = x + mlp(ln2(x))`。MLP：`Linear(d,4d) → GELU → Linear(4d,d)`。注意力必须是因果的（用 `-inf` 遮蔽未来）。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
block = {fn}(d_model=64, num_heads=4)
assert isinstance(block, nn.Module), 'GPT2Block should inherit from nn.Module'
out = block(torch.randn(2, 8, 64))
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Has LayerNorm (pre-norm architecture)",
            "code": """
import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'ln1') and isinstance(block.ln1, nn.LayerNorm), 'Need self.ln1 = nn.LayerNorm'
assert hasattr(block, 'ln2') and isinstance(block.ln2, nn.LayerNorm), 'Need self.ln2 = nn.LayerNorm'
""",
        },
        {
            "name": "MLP has 4x expansion with GELU",
            "code": """
import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'mlp'), 'Need self.mlp'
linears = [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]
assert len(linears) >= 2, f'MLP needs >= 2 Linear layers, got {len(linears)}'
assert linears[0].weight.shape == (128, 32), f'MLP first layer: {linears[0].weight.shape}, expected (128, 32)'
assert linears[-1].weight.shape == (32, 128), f'MLP last layer: {linears[-1].weight.shape}, expected (32, 128)'
""",
        },
        {
            "name": "Causal masking — future doesn't affect past",
            "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 8, 32)
out1 = block(x)
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 4, 32)
out2 = block(x2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Future tokens affected past — not causal'
""",
        },
        {
            "name": "Gradient flow to all parameters",
            "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 4, 32, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in block.parameters())
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
        },
    ],
    "solution": '''class _GELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            _GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _attn(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S, -1))

    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x''',
    "demo": """block = GPT2Block(64, 4)
print('Output:', block(torch.randn(2, 8, 64)).shape)
print('Params:', sum(p.numel() for p in block.parameters()))""",

}