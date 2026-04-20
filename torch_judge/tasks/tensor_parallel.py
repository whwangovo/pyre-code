"""Tensor Parallel MLP task."""

TASK = {
    "title": "Tensor Parallel MLP",
    "title_zh": "张量并行 MLP",
    "difficulty": "Hard",
    "description_en": "Implement a Megatron-style tensor parallel MLP.\n\nTensor parallelism splits the MLP weight matrices across devices. The first (up-projection) layer is column-parallel: each shard computes a slice of the hidden dimension. The second (down-projection) is row-parallel: each shard takes a slice of the input and produces the full output, then results are summed (all-reduce).\n\n**Signature:** `TensorParallelMLP(d_model, d_ff, world_size)` (nn.Module)\n\n**Parameters:**\n- `d_model` — input/output dimension\n- `d_ff` — hidden dimension (divisible by world_size)\n- `world_size` — number of virtual shards\n\n**Forward:** `forward(x) -> Tensor` where x is `(B, d_model)`\n\n**Constraint:** Store sharded weights as `nn.ParameterList`. The forward pass must simulate column-parallel + row-parallel + all-reduce. Result must match a standard `d_model -> d_ff -> d_model` MLP with GELU activation.",
    "description_zh": "实现 Megatron 风格的张量并行 MLP。\n\n张量并行将 MLP 权重矩阵分片到各设备。第一层（上投影）是列并行：每个分片计算隐藏维度的一个切片。第二层（下投影）是行并行：每个分片接收输入的一个切片并产生完整输出，然后求和（all-reduce）。\n\n**签名:** `TensorParallelMLP(d_model, d_ff, world_size)`（nn.Module）\n\n**参数:**\n- `d_model` — 输入/输出维度\n- `d_ff` — 隐藏维度（可被 world_size 整除）\n- `world_size` — 虚拟分片数\n\n**前向:** `forward(x) -> Tensor`，x 形状 `(B, d_model)`\n\n**约束:** 将分片权重存为 `nn.ParameterList`。前向传播必须模拟列并行 + 行并行 + all-reduce。结果必须与带 GELU 激活的标准 `d_model -> d_ff -> d_model` MLP 一致。",
    "function_name": "TensorParallelMLP",
    "hint": "1. W1: column-split → w1_shards[i] shape `(d_model, d_ff/world_size)`\n2. W2: row-split → w2_shards[i] shape `(d_ff/world_size, d_model)`\n3. for each shard i: h_i = gelu(x @ w1_i) → partial_i = h_i @ w2_i\n4. all-reduce = sum(partial_i for all i)",
    "hint_zh": "1. W1 按列分片 → w1_shards[i] 形状 `(d_model, d_ff/world_size)`\n2. W2 按行分片 → w2_shards[i] 形状 `(d_ff/world_size, d_model)`\n3. 对每个分片 i：h_i = gelu(x @ w1_i) → partial_i = h_i @ w2_i\n4. all-reduce = sum(所有 partial_i)",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(0)
model = {fn}(16, 32, world_size=2)
x = torch.randn(4, 16)
out = model(x)
assert out.shape == (4, 16), f'Expected (4, 16), got {out.shape}'
""",
        },
        {
            "name": "Matches standard MLP numerically",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(0)
d_model, d_ff, world_size = 16, 32, 4
tp_mlp = {fn}(d_model, d_ff, world_size)

# Build equivalent standard MLP by concatenating shards
W1_full = torch.cat([p.data for p in tp_mlp.w1_shards], dim=1)  # (d_model, d_ff)
W2_full = torch.cat([p.data for p in tp_mlp.w2_shards], dim=0)  # (d_ff, d_model)

torch.manual_seed(42)
x = torch.randn(3, d_model)
out_tp = tp_mlp(x)
out_std = torch.nn.functional.gelu(x @ W1_full) @ W2_full
assert torch.allclose(out_tp, out_std, atol=1e-5), f'TP MLP should match standard MLP'
""",
        },
        {
            "name": "world_size=1 works",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(8, 16, world_size=1)
x = torch.randn(2, 8)
out = model(x)
assert out.shape == (2, 8)
""",
        },
        {
            "name": "Gradient flows through all shards",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(8, 16, world_size=2)
x = torch.randn(2, 8, requires_grad=True)
out = model(x)
out.sum().backward()
assert x.grad is not None, 'No gradient for input'
for p in model.parameters():
    assert p.grad is not None, f'No gradient for parameter {p.shape}'
""",
        },
    ],
    "solution": '''class TensorParallelMLP(nn.Module):
    def __init__(self, d_model, d_ff, world_size):
        super().__init__()
        self.world_size = world_size
        shard_size = d_ff // world_size
        # Column-parallel: W1 shards (d_model, shard_size) each
        self.w1_shards = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, shard_size) * (2 / d_model) ** 0.5)
            for _ in range(world_size)
        ])
        # Row-parallel: W2 shards (shard_size, d_model) each
        self.w2_shards = nn.ParameterList([
            nn.Parameter(torch.randn(shard_size, d_model) * (2 / d_ff) ** 0.5)
            for _ in range(world_size)
        ])

    def forward(self, x):
        # Column-parallel forward + row-parallel + all-reduce (sum)
        output = None
        for w1, w2 in zip(self.w1_shards, self.w2_shards):
            z = x @ w1
            h = z * 0.5 * (1.0 + torch.erf(z / (2.0 ** 0.5)))  # GELU
            partial = h @ w2                        # (B, d_model)
            output = partial if output is None else output + partial
        return output''',
    "demo": """torch.manual_seed(42)
d_model, d_ff, world_size = 64, 256, 4
x = torch.randn(2, d_model)

tp_mlp = TensorParallelMLP(d_model, d_ff, world_size)
out = tp_mlp(x)
print("Output shape:", out.shape)  # expect (2, 64)

w1_full = torch.cat([p.data for p in tp_mlp.w1_shards], dim=1)  # (d_model, d_ff)
w2_full = torch.cat([p.data for p in tp_mlp.w2_shards], dim=0)  # (d_ff, d_model)
ref = torch.nn.functional.gelu(x @ w1_full) @ w2_full

print("Max diff vs reference:", (out - ref).abs().max().item())  # expect ~0""",

}
