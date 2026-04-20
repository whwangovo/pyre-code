"""Multi-Head Latent Attention (MLA) task."""

TASK = {
    "title": "Multi-Head Latent Attention (MLA)",
    "title_zh": "多头潜在注意力（MLA）",
    "difficulty": "Hard",
    "description_en": "Implement Multi-Head Latent Attention (MLA) from DeepSeek V2/V3.\n\nMLA's key innovation: instead of caching full K and V tensors, compress them into a low-rank latent vector `c_kv`, then decompress on the fly. This dramatically reduces KV cache memory during inference.\n\n**Signature:** `mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads) -> Tensor`\n\n**Parameters:**\n- `X` — input tensor (B, S, D)\n- `W_dkv` — KV compression matrix (D, kv_rank)\n- `W_uk` — K decompression matrix (kv_rank, num_heads * D_h)\n- `W_uv` — V decompression matrix (kv_rank, num_heads * D_h)\n- `W_q` — Q projection matrix (D, num_heads * D_h)\n- `num_heads` — number of attention heads\n\n**Returns:** output tensor (B, S, num_heads * D_h)\n\n**Steps:**\n1. Compress: `c_kv = X @ W_dkv` → (B, S, kv_rank)\n2. Decompress: `K = c_kv @ W_uk`, `V = c_kv @ W_uv`\n3. Project queries: `Q = X @ W_q`\n4. Reshape all to (B, num_heads, S, D_h)\n5. Scaled dot-product attention\n6. Reshape output to (B, S, num_heads * D_h)",
    "description_zh": "实现 DeepSeek V2/V3 中的多头潜在注意力（MLA）。\n\nMLA 的核心创新：不缓存完整的 K 和 V 张量，而是将其压缩为低秩潜在向量 `c_kv`，推理时再即时解压。这大幅降低了推理时的 KV 缓存内存占用。\n\n**签名:** `mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads) -> Tensor`\n\n**参数:**\n- `X` — 输入张量 (B, S, D)\n- `W_dkv` — KV 压缩矩阵 (D, kv_rank)\n- `W_uk` — K 解压矩阵 (kv_rank, num_heads * D_h)\n- `W_uv` — V 解压矩阵 (kv_rank, num_heads * D_h)\n- `W_q` — Q 投影矩阵 (D, num_heads * D_h)\n- `num_heads` — 注意力头数\n\n**返回:** 输出张量 (B, S, num_heads * D_h)\n\n**步骤:**\n1. 压缩：`c_kv = X @ W_dkv` → (B, S, kv_rank)\n2. 解压：`K = c_kv @ W_uk`，`V = c_kv @ W_uv`\n3. 投影查询：`Q = X @ W_q`\n4. 重塑为 (B, num_heads, S, D_h)\n5. 缩放点积注意力\n6. 重塑输出为 (B, S, num_heads * D_h)",
    "function_name": "mla_attention",
    "hint": "1. `c_kv = X @ W_dkv`  → `(B,S,kv_rank)`\n2. `K = c_kv @ W_uk`, `V = c_kv @ W_uv`, `Q = X @ W_q`\n3. `.view(B,S,num_heads,D_h).transpose(1,2)` for each\n4. Scaled dot-product attn → `.transpose(1,2).reshape(B,S,-1)`",
    "hint_zh": "1. `c_kv = X @ W_dkv`  → `(B,S,kv_rank)`\n2. `K = c_kv @ W_uk`，`V = c_kv @ W_uv`，`Q = X @ W_q`\n3. 各自 `.view(B,S,num_heads,D_h).transpose(1,2)`\n4. 缩放点积注意力 → `.transpose(1,2).reshape(B,S,-1)`",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
B, S, D, kv_rank, num_heads, D_h = 2, 6, 64, 16, 4, 8
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
assert out.shape == (B, S, num_heads * D_h), f'Expected ({B}, {S}, {num_heads * D_h}), got {out.shape}'
"""
        },
        {
            "name": "KV latent has correct compressed shape",
            "code": """
import torch
torch.manual_seed(1)
B, S, D, kv_rank, num_heads, D_h = 1, 8, 32, 6, 2, 4
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
# Conceptual check: verify the compression dimensions are correct.
# This independently computes c_kv to validate the shape of the latent
# representation. The student's actual code path is authoritatively
# validated by the 'Numerical correctness' test (test 5).
c_kv = X @ W_dkv
assert c_kv.shape == (B, S, kv_rank), f'Latent shape should be ({B}, {S}, {kv_rank}), got {c_kv.shape}'
assert kv_rank < D, 'kv_rank should be smaller than D for compression'
assert out.shape == (B, S, num_heads * D_h), f'Output shape mismatch: {out.shape}'
"""
        },
        {
            "name": "Gradient flows through X and weights",
            "code": """
import torch
torch.manual_seed(2)
B, S, D, kv_rank, num_heads, D_h = 1, 4, 16, 4, 2, 4
W_dkv = torch.randn(D, kv_rank, requires_grad=True)
W_uk  = torch.randn(kv_rank, num_heads * D_h, requires_grad=True)
W_uv  = torch.randn(kv_rank, num_heads * D_h, requires_grad=True)
W_q   = torch.randn(D, num_heads * D_h, requires_grad=True)
X = torch.randn(B, S, D, requires_grad=True)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
out.sum().backward()
assert X.grad is not None, 'Missing gradient for X'
assert W_dkv.grad is not None, 'Missing gradient for W_dkv'
assert W_uk.grad is not None, 'Missing gradient for W_uk'
assert W_uv.grad is not None, 'Missing gradient for W_uv'
assert W_q.grad is not None, 'Missing gradient for W_q'
"""
        },
        {
            "name": "kv_rank compression reduces memory",
            "code": """
import torch
torch.manual_seed(3)
B, S, D, kv_rank, num_heads, D_h = 2, 10, 128, 8, 4, 16
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
# c_kv should be much smaller than full KV
c_kv = X @ W_dkv
full_kv_size = B * S * num_heads * D_h * 2  # K and V
compressed_size = B * S * kv_rank
assert c_kv.shape == (B, S, kv_rank), f'Wrong latent shape: {c_kv.shape}'
assert compressed_size < full_kv_size, 'Compressed KV should be smaller than full KV'
assert out.shape == (B, S, num_heads * D_h), f'Output shape mismatch: {out.shape}'
"""
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch
torch.manual_seed(9)
B, S, D, kv_rank, num_heads, D_h = 2, 5, 32, 8, 4, 8
W_dkv = torch.randn(D, kv_rank)
W_uk  = torch.randn(kv_rank, num_heads * D_h)
W_uv  = torch.randn(kv_rank, num_heads * D_h)
W_q   = torch.randn(D, num_heads * D_h)
X = torch.randn(B, S, D)
out = {fn}(X, W_dkv, W_uk, W_uv, W_q, num_heads)
# Reference computation
c_kv = X @ W_dkv                          # (B, S, kv_rank)
K = c_kv @ W_uk                            # (B, S, num_heads*D_h)
V = c_kv @ W_uv                            # (B, S, num_heads*D_h)
Q = X @ W_q                                # (B, S, num_heads*D_h)
def split_heads(t):
    return t.view(B, S, num_heads, D_h).transpose(1, 2)
Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
scale = D_h ** -0.5
attn = torch.softmax(Qh @ Kh.transpose(-2, -1) * scale, dim=-1)
expected = (attn @ Vh).transpose(1, 2).reshape(B, S, num_heads * D_h)
assert torch.allclose(out, expected, atol=1e-5), f'MLA numerical mismatch: max diff {(out - expected).abs().max()}'
"""
        }
    ],
    "solution": '''def mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads):
    B, S, D = X.shape
    D_h = W_q.shape[1] // num_heads
    # Compress KV into low-rank latent
    c_kv = X @ W_dkv                          # (B, S, kv_rank)
    K = c_kv @ W_uk                            # (B, S, num_heads*D_h)
    V = c_kv @ W_uv                            # (B, S, num_heads*D_h)
    Q = X @ W_q                                # (B, S, num_heads*D_h)
    # Reshape to multi-head format
    def split_heads(t):
        return t.view(B, S, num_heads, D_h).transpose(1, 2)
    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
    scale = D_h ** -0.5
    attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    out = (attn @ V).transpose(1, 2).reshape(B, S, num_heads * D_h)
    return out''',
    "demo": """torch.manual_seed(0)
B, S, D = 2, 6, 32
num_heads = 4
D_h = 8          # head dim
D_c = 8          # compressed KV dim (latent)

W_dkv = torch.randn(D, D_c) * 0.1      # compress to latent
W_uk  = torch.randn(D_c, num_heads * D_h) * 0.1  # up-project to K
W_uv  = torch.randn(D_c, num_heads * D_h) * 0.1  # up-project to V
W_q   = torch.randn(D, num_heads * D_h) * 0.1

X = torch.randn(B, S, D)

c_kv = X @ W_dkv
K_full = c_kv @ W_uk
print(f"Input shape:          {X.shape}")       # (2, 6, 32)
print(f"Compressed KV shape:  {c_kv.shape}")    # (2, 6, 8)  <-- small latent
print(f"Full K shape:         {K_full.shape}")  # (2, 6, 32) <-- expanded

out = mla_attention(X, W_dkv, W_uk, W_uv, W_q, num_heads)
print(f"Output shape:         {out.shape}")     # (2, 6, 32)""",

}
