"""Depthwise Separable Convolution task."""

TASK = {
    "title": "Depthwise Separable Convolution",
    "title_zh": "深度可分离卷积",
    "difficulty": "Medium",
    "description_en": "Implement MobileNet-style depthwise separable convolution from primitives.\n\nDepthwise separable convolution factorizes a standard convolution into two steps: a depthwise conv (one filter per input channel) followed by a pointwise 1x1 conv (to mix channels). This dramatically reduces parameter count and FLOPs.\n\n**Signature:** `depthwise_separable_conv(x, dw_weight, pw_weight) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor (B, C_in, H, W)\n- `dw_weight` — depthwise filter (C_in, 1, kH, kW)\n- `pw_weight` — pointwise filter (C_out, C_in, 1, 1)\n\n**Returns:** output tensor (B, C_out, H-kH+1, W-kW+1)\n\n**Constraints:**\n- No padding, stride=1\n- Must NOT use `F.conv2d` — implement both steps using `unfold` and `einsum`/matmul\n- Depthwise step: each channel c is convolved only with `dw_weight[c, 0]`\n- Pointwise step: 1x1 conv mixes channels via matrix multiply",
    "description_zh": "从基本原语实现 MobileNet 风格的深度可分离卷积。\n\n深度可分离卷积将标准卷积分解为两步：深度卷积（每个输入通道一个滤波器）和逐点 1x1 卷积（混合通道）。这大幅减少了参数量和计算量。\n\n**签名:** `depthwise_separable_conv(x, dw_weight, pw_weight) -> Tensor`\n\n**参数:**\n- `x` — 输入张量 (B, C_in, H, W)\n- `dw_weight` — 深度滤波器 (C_in, 1, kH, kW)\n- `pw_weight` — 逐点滤波器 (C_out, C_in, 1, 1)\n\n**返回:** 输出张量 (B, C_out, H-kH+1, W-kW+1)\n\n**约束:**\n- 无填充，步长=1\n- 不得使用 `F.conv2d`——使用 `unfold` 和 `einsum`/matmul 实现两步\n- 深度步骤：通道 c 仅与 `dw_weight[c, 0]` 卷积\n- 逐点步骤：1x1 卷积通过矩阵乘法混合通道",
    "function_name": "depthwise_separable_conv",
    "hint": "Depthwise: `x.unfold(2,kH,1).unfold(3,kW,1)` → `(B,C,H_out,W_out,kH,kW)` → multiply `dw_weight` → sum last 2 dims\nPointwise: `torch.einsum('bchw,oc->bohw', dw_out, pw_weight[:,: ,0,0])`\n",
    "hint_zh": "深度卷积：`x.unfold(2,kH,1).unfold(3,kW,1)` → `(B,C,H_out,W_out,kH,kW)` → 乘 `dw_weight` → 对最后 2 维求和\n逐点卷积：`torch.einsum('bchw,oc->bohw', dw_out, pw_weight[:,: ,0,0])`",
    "tests": [
        {
            "name": "Output shape is correct",
            "code": """
import torch
torch.manual_seed(0)
B, C_in, H, W = 2, 4, 8, 8
kH, kW, C_out = 3, 3, 6
x = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)
pw_weight = torch.randn(C_out, C_in, 1, 1)
out = {fn}(x, dw_weight, pw_weight)
expected_shape = (B, C_out, H - kH + 1, W - kW + 1)
assert out.shape == expected_shape, f'Shape: {out.shape}, expected {expected_shape}'
""",
        },
        {
            "name": "Depthwise step: channel independence",
            "code": """
import torch
torch.manual_seed(1)
B, C_in, H, W = 1, 3, 6, 6
kH, kW = 3, 3
x = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)
pw_weight = torch.eye(C_in).unsqueeze(-1).unsqueeze(-1)  # identity pointwise
out1 = {fn}(x, dw_weight, pw_weight)
x2 = x.clone()
x2[:, 1] = x2[:, 1] * 0  # zero out channel 1
out2 = {fn}(x2, dw_weight, pw_weight)
# channel 0 output should be unchanged
assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Channel 0 output changed when channel 1 was zeroed — depthwise not independent'
""",
        },
        {
            "name": "Matches reference loop implementation",
            "code": """
import torch
torch.manual_seed(2)
B, C_in, H, W = 1, 2, 5, 5
kH, kW, C_out = 3, 3, 3
x = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)
pw_weight = torch.randn(C_out, C_in, 1, 1)
H_out, W_out = H - kH + 1, W - kW + 1
# Reference: loop-based depthwise then pointwise
dw_ref = torch.zeros(B, C_in, H_out, W_out)
for c in range(C_in):
    for i in range(H_out):
        for j in range(W_out):
            dw_ref[:, c, i, j] = (x[:, c, i:i+kH, j:j+kW] * dw_weight[c, 0]).sum(dim=(-2, -1))
ref = torch.einsum('bchw,oc->bohw', dw_ref, pw_weight[:, :, 0, 0])
out = {fn}(x, dw_weight, pw_weight)
assert torch.allclose(out, ref, atol=1e-5), f'Max diff: {(out - ref).abs().max().item()}'
""",
        },
        {
            "name": "Gradient flows",
            "code": """
import torch
torch.manual_seed(3)
x = torch.randn(1, 2, 6, 6, requires_grad=True)
dw_weight = torch.randn(2, 1, 3, 3, requires_grad=True)
pw_weight = torch.randn(4, 2, 1, 1, requires_grad=True)
out = {fn}(x, dw_weight, pw_weight)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert dw_weight.grad is not None, 'dw_weight.grad is None'
assert pw_weight.grad is not None, 'pw_weight.grad is None'
""",
        },
    ],
    "solution": """def depthwise_separable_conv(x, dw_weight, pw_weight):
    B, C_in, H, W = x.shape
    kH, kW = dw_weight.shape[2], dw_weight.shape[3]
    H_out, W_out = H - kH + 1, W - kW + 1
    # Depthwise: unfold spatial dims to extract patches
    patches = x.unfold(2, kH, 1).unfold(3, kW, 1)  # (B, C_in, H_out, W_out, kH, kW)
    dw_out = (patches * dw_weight[:, 0].view(1, C_in, 1, 1, kH, kW)).sum(dim=(-2, -1))  # (B, C_in, H_out, W_out)
    # Pointwise: 1x1 conv = channel-wise linear combination
    out = torch.einsum('bchw,oc->bohw', dw_out, pw_weight[:, :, 0, 0])
    return out""",
    "demo": """torch.manual_seed(0)
B, C_in, H, W = 2, 4, 8, 8
C_out, kH, kW = 8, 3, 3

x         = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)   # one kernel per input channel
pw_weight = torch.randn(C_out, C_in, 1, 1)  # 1x1 conv

out = depthwise_separable_conv(x, dw_weight, pw_weight)
print("Output shape:", out.shape)  # (2, 8, 6, 6)

patches = x.unfold(2, kH, 1).unfold(3, kW, 1)
dw_ch0 = (patches[:, 0:1] * dw_weight[0:1, 0].view(1, 1, 1, 1, kH, kW)).sum(dim=(-2, -1))
dw_ch1 = (patches[:, 1:2] * dw_weight[1:2, 0].view(1, 1, 1, 1, kH, kW)).sum(dim=(-2, -1))
print("DW ch0 and ch1 are independent (cross-correlation ~0):",
      (dw_ch0 * dw_ch1).mean().abs().item() < 1.0)""",

}
