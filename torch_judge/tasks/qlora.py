"""QLoRA (Quantized Low-Rank Adaptation) task."""

TASK = {
    "title": "QLoRA",
    "title_zh": "QLoRA",
    "difficulty": "Hard",
    "description_en": "Implement QLoRA: a quantized linear layer with LoRA adapters.\n\nQLoRA stores the base weight in int8 (simulating 4-bit quantization) with a per-row fp32 scale, while LoRA adapters remain in full precision for efficient fine-tuning.\n\n**Signature:** `QLoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)\n\n**Parameters stored:**\n- `quantized_weight`: (out_features, in_features), dtype=int8, requires_grad=False\n- `scale`: (out_features, 1), fp32 per-row scale\n- `lora_A`: (in_features, rank), initialized ~ N(0, 0.01²)\n- `lora_B`: (rank, out_features), initialized to zeros\n\n**Forward:**\n1. Dequantize: `W_fp = quantized_weight.float() * scale`\n2. Base output: `x @ W_fp.T`\n3. LoRA delta: `x @ lora_A @ lora_B * (alpha / rank)`\n4. Return base + delta\n\n**Helper:** `set_weight(W_fp32)` quantizes a float weight per-row:\n- `scale = W.abs().max(dim=1, keepdim=True).values / 127`\n- `quantized = (W / scale).round().clamp(-127, 127).to(int8)`",
    "description_zh": "实现 QLoRA：带 LoRA 适配器的量化线性层。\n\nQLoRA 将基础权重以 int8 格式存储（模拟 4-bit 量化），配合每行 fp32 缩放因子，同时 LoRA 适配器保持全精度，实现高效微调。\n\n**签名:** `QLoRALinear(in_features, out_features, rank, alpha=1.0)`（nn.Module）\n\n**存储的参数:**\n- `quantized_weight`：(out_features, in_features)，dtype=int8，requires_grad=False\n- `scale`：(out_features, 1)，fp32 每行缩放因子\n- `lora_A`：(in_features, rank)，初始化 ~ N(0, 0.01²)\n- `lora_B`：(rank, out_features)，初始化为零\n\n**前向传播:**\n1. 反量化：`W_fp = quantized_weight.float() * scale`\n2. 基础输出：`x @ W_fp.T`\n3. LoRA 增量：`x @ lora_A @ lora_B * (alpha / rank)`\n4. 返回 base + delta\n\n**辅助方法:** `set_weight(W_fp32)` 按行量化浮点权重：\n- `scale = W.abs().max(dim=1, keepdim=True).values / 127`\n- `quantized = (W / scale).round().clamp(-127, 127).to(int8)`",
    "function_name": "QLoRALinear",
    "hint": "set_weight: `scale = W.abs().max(dim=1,keepdim=True).values.clamp(1e-8)/127`\n           `q = (W/scale).round().clamp(-127,127).to(int8)`\nForward: `W_fp = q.float()*scale` → `x@W_fp.T + x@lora_A@lora_B*(alpha/rank)`",
    "hint_zh": "set_weight：`scale = W.abs().max(dim=1,keepdim=True).values.clamp(1e-8)/127`\n            `q = (W/scale).round().clamp(-127,127).to(int8)`\n前向：`W_fp = q.float()*scale` → `x@W_fp.T + x@lora_A@lora_B*(alpha/rank)`",
    "tests": [
        {
            "name": "Output shape correct",
            "code": "\nimport torch, torch.nn as nn\nlayer = {fn}(in_features=16, out_features=8, rank=4)\nassert isinstance(layer, nn.Module)\nx = torch.randn(3, 16)\nout = layer(x)\nassert out.shape == (3, 8), f'Expected (3, 8), got {out.shape}'\n"
        },
        {
            "name": "Dequantized weight close to original",
            "code": "\nimport torch\ntorch.manual_seed(42)\nlayer = {fn}(in_features=16, out_features=8, rank=4)\nW = torch.randn(8, 16)\nlayer.set_weight(W)\nW_fp = layer.quantized_weight.float() * layer.scale\nassert torch.allclose(W, W_fp, atol=0.02), f'Max dequant error: {(W - W_fp).abs().max().item():.4f}'\n"
        },
        {
            "name": "LoRA delta is zero when lora_B is zero",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlayer = {fn}(in_features=8, out_features=4, rank=2)\nW = torch.randn(4, 8)\nlayer.set_weight(W)\nx = torch.randn(2, 8)\nW_fp = layer.quantized_weight.float() * layer.scale\nbase = x @ W_fp.T\nout = layer(x)\nassert torch.allclose(out, base, atol=1e-5), 'With lora_B=0, output should equal base'\n"
        },
        {
            "name": "Gradient flows through LoRA, not quantized_weight",
            "code": "\nimport torch\nlayer = {fn}(in_features=8, out_features=4, rank=2)\nlayer(torch.randn(2, 8)).sum().backward()\nassert layer.lora_A.grad is not None, 'lora_A should have grad'\nassert layer.lora_B.grad is not None, 'lora_B should have grad'\nassert layer.quantized_weight.grad is None, 'quantized_weight must not have grad'\n"
        },
        {
            "name": "set_weight then forward close to full-precision linear",
            "code": "\nimport torch\ntorch.manual_seed(7)\nlayer = {fn}(in_features=16, out_features=8, rank=4)\nW = torch.randn(8, 16)\nlayer.set_weight(W)\n# zero out LoRA so we isolate the quantized base\nlayer.lora_B.data.zero_()\nx = torch.randn(5, 16)\nout = layer(x)\nref = x @ W.T\nassert torch.allclose(out, ref, atol=0.05), f'Max error vs fp32: {(out - ref).abs().max().item():.4f}'\n"
        }
    ],
    "solution": '''class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.quantized_weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def set_weight(self, W_fp32):
        scale = W_fp32.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
        q = (W_fp32 / scale).round().clamp(-127, 127).to(torch.int8)
        self.quantized_weight.data.copy_(q)
        self.scale.data.copy_(scale)

    def forward(self, x):
        W_fp = self.quantized_weight.float() * self.scale
        base = x @ W_fp.T
        delta = x @ self.lora_A @ self.lora_B * (self.alpha / self.rank)
        return base + delta''',
    "demo": """torch.manual_seed(0)
in_f, out_f, rank = 64, 32, 4
layer = QLoRALinear(in_f, out_f, rank, alpha=2.0)

W_ref = torch.randn(out_f, in_f)
layer.set_weight(W_ref)

x = torch.randn(8, in_f)
y_qlora = layer(x)
y_ref = x @ W_ref.T  # full-precision baseline (no LoRA delta)

print("Output shape:", y_qlora.shape)          # (8, 32)
print("Max abs error vs fp32:", (y_qlora - y_ref).abs().max().item())""",

}
