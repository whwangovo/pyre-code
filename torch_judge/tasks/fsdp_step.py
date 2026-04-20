"""FSDP Training Step task."""

TASK = {
    "title": "FSDP Training Step",
    "title_zh": "FSDP 训练步骤",
    "difficulty": "Hard",
    "description_en": "Implement a simplified FSDP (Fully Sharded Data Parallel) training step.\n\nFSDP shards model parameters across workers to reduce per-device memory. Before the forward pass, each worker all-gathers the full parameters. After the backward pass, gradients are reduce-scattered so each worker only holds its shard's gradient.\n\n**Signature:** `fsdp_step(param_shards, grad_fn, world_size) -> list[Tensor]`\n\n**Parameters:**\n- `param_shards` — list of `world_size` parameter shards (each a flat 1D tensor)\n- `grad_fn` — callable that takes the full (unsharded) parameter tensor and returns the gradient w.r.t. it\n- `world_size` — number of virtual workers\n\n**Returns:** updated list of parameter shards (after one gradient step with lr=0.01)\n\n**Steps:**\n1. All-gather: concatenate shards to get full parameter\n2. Compute gradient via grad_fn(full_param)\n3. Reduce-scatter: split gradient into world_size chunks, each worker keeps its chunk\n4. Update each shard: shard -= 0.01 * grad_shard",
    "description_zh": "实现简化版 FSDP（全分片数据并行）训练步骤。\n\nFSDP 将模型参数分片到各 worker 以减少单设备内存。前向传播前，每个 worker all-gather 完整参数。反向传播后，梯度 reduce-scatter，每个 worker 只保留自己分片的梯度。\n\n**签名:** `fsdp_step(param_shards, grad_fn, world_size) -> list[Tensor]`\n\n**参数:**\n- `param_shards` — `world_size` 个参数分片的列表（每个是扁平 1D 张量）\n- `grad_fn` — 接收完整（未分片）参数张量，返回对应梯度的可调用对象\n- `world_size` — 虚拟 worker 数量\n\n**返回:** 更新后的参数分片列表（一步梯度更新，lr=0.01）\n\n**步骤:**\n1. All-gather：拼接分片得到完整参数\n2. 通过 grad_fn(full_param) 计算梯度\n3. Reduce-scatter：将梯度分成 world_size 块，每个 worker 保留自己的块\n4. 更新每个分片：shard -= 0.01 * grad_shard",
    "function_name": "fsdp_step",
    "hint": "1. all-gather: full_param = torch.cat(param_shards)\n2. grad = grad_fn(full_param)\n3. reduce-scatter: grad_shards = list(grad.chunk(world_size))\n4. new_shards[i] = param_shards[i] - 0.01 * grad_shards[i]",
    "hint_zh": "1. all-gather：full_param = torch.cat(param_shards)\n2. grad = grad_fn(full_param)\n3. reduce-scatter：grad_shards = list(grad.chunk(world_size))\n4. new_shards[i] = param_shards[i] - 0.01 * grad_shards[i]",
    "tests": [
        {
            "name": "Returns correct number of shards",
            "code": """
import torch
world_size = 4
shard_size = 8
param_shards = [torch.randn(shard_size) for _ in range(world_size)]
def grad_fn(p):
    return torch.ones_like(p)
new_shards = {fn}(param_shards, grad_fn, world_size)
assert len(new_shards) == world_size, f'Expected {world_size} shards, got {len(new_shards)}'
assert all(s.shape == (shard_size,) for s in new_shards), 'Shard shapes should be preserved'
""",
        },
        {
            "name": "Matches equivalent non-sharded SGD update",
            "code": """
import torch
torch.manual_seed(0)
world_size = 2
shard_size = 4
param_shards = [torch.randn(shard_size) for _ in range(world_size)]
full_param_ref = torch.cat(param_shards).clone()

# grad_fn: gradient is 2 * full_param (from loss = ||param||^2)
def grad_fn(p):
    return 2.0 * p

new_shards = {fn}(param_shards, grad_fn, world_size)
new_full = torch.cat(new_shards)

# Expected: SGD step on full param
grad_ref = grad_fn(full_param_ref)
expected = full_param_ref - 0.01 * grad_ref
assert torch.allclose(new_full, expected, atol=1e-6), f'FSDP step should match SGD on full param'
""",
        },
        {
            "name": "Each shard gets its own gradient slice",
            "code": """
import torch
torch.manual_seed(1)
world_size = 4
shard_size = 4
param_shards = [torch.randn(shard_size) for _ in range(world_size)]
full_param = torch.cat(param_shards)

# Gradient is position-dependent: grad[i] = i
def grad_fn(p):
    return torch.arange(len(p), dtype=p.dtype)

new_shards = {fn}(param_shards, grad_fn, world_size)
full_grad = grad_fn(full_param)
grad_chunks = list(full_grad.chunk(world_size))
for i, (shard, orig, gc) in enumerate(zip(new_shards, param_shards, grad_chunks)):
    expected = orig - 0.01 * gc
    assert torch.allclose(shard, expected, atol=1e-6), f'Shard {i} update incorrect'
""",
        },
        {
            "name": "Works with world_size=1",
            "code": """
import torch
torch.manual_seed(2)
param_shards = [torch.randn(8)]
def grad_fn(p):
    return p * 2
new_shards = {fn}(param_shards, grad_fn, world_size=1)
assert len(new_shards) == 1
expected = param_shards[0] - 0.01 * grad_fn(param_shards[0])
assert torch.allclose(new_shards[0], expected, atol=1e-6)
""",
        },
    ],
    "solution": '''def fsdp_step(param_shards, grad_fn, world_size):
    # 1. All-gather: reconstruct full parameter
    full_param = torch.cat(param_shards)

    # 2. Compute gradient
    grad = grad_fn(full_param)

    # 3. Reduce-scatter: each worker gets its gradient shard
    grad_shards = list(grad.chunk(world_size))

    # 4. Update each shard with lr=0.01
    new_shards = [
        param_shards[i] - 0.01 * grad_shards[i]
        for i in range(world_size)
    ]
    return new_shards''',
    "demo": """torch.manual_seed(0)
world_size = 4
shard_size = 8

param_shards = [torch.randn(shard_size) for _ in range(world_size)]

grad_fn = lambda p: 2.0 * p

new_shards = fsdp_step(param_shards, grad_fn, world_size)
fsdp_result = torch.cat(new_shards)

full_param = torch.cat(param_shards)
ref_result = full_param - 0.01 * grad_fn(full_param)

print("FSDP result shape:", fsdp_result.shape)  # expect (32,)
print("Max diff vs SGD reference:", (fsdp_result - ref_result).abs().max().item())  # expect ~0""",

}
