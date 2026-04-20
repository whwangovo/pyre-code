"""INT8 Quantized Linear task."""

TASK = {
    "title": "INT8 Quantized Linear",
    "title_zh": "INT8 量化线性层",
    "difficulty": "Hard",
    "description_en": "Implement INT8 per-channel weight quantization for a linear layer.\n\nQuantization converts float32 weights to int8 with per-channel scaling, reducing model size by 4x while preserving accuracy.\n\n**Signature:** `Int8Linear(weight, bias=None)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — input tensor (*, in_features)\n\n**Returns:** linear output with dequantized weights\n\n**Constraints:**\n- Per-channel scale: `abs(weight).max(dim=1) / 127`\n- Quantize: `round(weight/scale).clamp(-128, 127).to(int8)`\n- Store weight_int8 and scale as buffers, not parameters",
    "description_zh": "实现 INT8 逐通道权重量化线性层。\n\n量化将 float32 权重转换为 int8 并使用逐通道缩放，将模型大小减少 4 倍同时保持精度。\n\n**签名:** `Int8Linear(weight, bias=None)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 输入张量 (*, in_features)\n\n**返回:** 使用反量化权重的线性输出\n\n**约束:**\n- 逐通道缩放：`abs(weight).max(dim=1) / 127`\n- 量化：`round(weight/scale).clamp(-128, 127).to(int8)`\n- weight_int8 和 scale 存储为 buffer 而非 parameter",
    "function_name": "Int8Linear",
    "hint": "1. scale = abs(weight).max(dim=1) / 127  (per output channel, keepdim)\n2. weight_int8 = round(weight / scale).clamp(-128, 127).to(int8)\n3. register both as buffers (not parameters)\n4. forward: dequant = weight_int8.float() * scale → x @ dequant.T",
    "hint_zh": "1. scale = abs(weight).max(dim=1) / 127（逐输出通道，keepdim）\n2. weight_int8 = round(weight / scale).clamp(-128, 127).to(int8)\n3. 两者均注册为 buffer（非 parameter）\n4. 前向：dequant = weight_int8.float() * scale → x @ dequant.T",
    "tests": [
        {
            "name": "Weight is int8",
            "code": "\nimport torch, torch.nn as nn\nw = torch.randn(32, 16)\nq = {fn}(w)\nassert isinstance(q, nn.Module)\nassert q.weight_int8.dtype == torch.int8, f'dtype: {q.weight_int8.dtype}'\n"
        },
        {
            "name": "Values in [-128, 127]",
            "code": "\nimport torch\nq = {fn}(torch.randn(64, 32) * 10)\nassert q.weight_int8.min() >= -128 and q.weight_int8.max() <= 127\n"
        },
        {
            "name": "Dequantized close to original",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.randn(16, 8)\nq = {fn}(w)\nw_recon = q.weight_int8.float() * q.scale\nassert (w - w_recon).abs().max() < 0.1, 'Quantization error too large'\n"
        },
        {
            "name": "Forward with bias correctness",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.randn(8, 4)\nb = torch.randn(8)\nq_bias = {fn}(w, b)\nq_no_bias = {fn}(w)\nx = torch.randn(2, 4)\nout_bias = q_bias(x)\nout_no = q_no_bias(x)\nassert out_bias.shape == (2, 8), f'Shape: {out_bias.shape}'\nassert torch.allclose(out_bias - out_no, b.unsqueeze(0).expand(2, -1), atol=1e-5), 'Bias not correctly added'\n"
        },
        {
            "name": "Weight is buffer not parameter",
            "code": "\nimport torch\nq = {fn}(torch.randn(4, 4))\nparam_names = [n for n, _ in q.named_parameters()]\nassert 'weight_int8' not in param_names, 'weight_int8 should be a buffer'\nassert 'scale' not in param_names, 'scale should be a buffer'\n"
        }
    ],
    "solution": '''class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        self.register_buffer('weight_int8',
            torch.round(weight / (scale + 1e-10)).clamp(-128, 127).to(torch.int8))
        self.register_buffer('scale', scale)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out''',
    "demo": """w = torch.randn(8, 4)
q = Int8Linear(w)
print('Output:', q(torch.randn(2, 4)).shape)
print('Weight dtype:', q.weight_int8.dtype)
print('Compression: float32 -> int8 = 4x')""",

}