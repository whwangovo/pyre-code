"""Sinusoidal Position Encoding task."""

TASK = {
    "title": "Sinusoidal Position Encoding",
    "title_zh": "正弦位置编码",
    "difficulty": "Easy",
    "description_en": "Implement the sinusoidal position encoding from 'Attention Is All You Need'.\n\nPosition encodings are added to token embeddings to give the model information about token positions. The original Transformer uses fixed sinusoidal functions.\n\n**Signature:** `sinusoidal_pe(seq_len, d_model) -> Tensor`\n\n**Parameters:**\n- `seq_len` — number of positions\n- `d_model` — embedding dimension (must be even)\n\n**Returns:** position encoding tensor of shape `(seq_len, d_model)`\n\n**Formula:**\n- `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`\n- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`",
    "description_zh": "实现《Attention Is All You Need》中的正弦位置编码。\n\n位置编码加到词嵌入上，让模型感知 token 的位置信息。原始 Transformer 使用固定的正弦函数。\n\n**签名:** `sinusoidal_pe(seq_len, d_model) -> Tensor`\n\n**参数:**\n- `seq_len` — 位置数量\n- `d_model` — 嵌入维度（必须为偶数）\n\n**返回:** 形状为 `(seq_len, d_model)` 的位置编码张量\n\n**公式:**\n- `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`\n- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`",
    "function_name": "sinusoidal_pe",
    "hint": "1. `freqs = 1 / 10000^(2i/d_model)` for i in `range(d_model//2)`\n2. `angles = pos[:, None] * freqs[None, :]` → shape `(seq_len, d_model//2)`\n3. `pe[:, 0::2] = sin(angles)`, `pe[:, 1::2] = cos(angles)`",
    "hint_zh": "1. `freqs = 1 / 10000^(2i/d_model)`，i 取 `range(d_model//2)`\n2. `angles = pos[:, None] * freqs[None, :]` → 形状 `(seq_len, d_model//2)`\n3. `pe[:, 0::2] = sin(angles)`，`pe[:, 1::2] = cos(angles)`",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
pe = {fn}(10, 16)
assert pe.shape == (10, 16), f'Expected (10, 16), got {pe.shape}'
""",
        },
        {
            "name": "Even columns are sin, odd are cos",
            "code": """
import torch
pe = {fn}(5, 8)
pos = torch.arange(5).float().unsqueeze(1)
freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
angles = pos * freq
assert torch.allclose(pe[:, 0::2], torch.sin(angles), atol=1e-5), 'Even columns should be sin'
assert torch.allclose(pe[:, 1::2], torch.cos(angles), atol=1e-5), 'Odd columns should be cos'
""",
        },
        {
            "name": "Position 0 has sin=0",
            "code": """
import torch
pe = {fn}(8, 32)
assert torch.allclose(pe[0, 0::2], torch.zeros(16), atol=1e-6), 'sin(0) should be 0'
assert torch.allclose(pe[0, 1::2], torch.ones(16), atol=1e-6), 'cos(0) should be 1'
""",
        },
        {
            "name": "Values in [-1, 1]",
            "code": """
import torch
pe = {fn}(100, 64)
assert pe.min() >= -1.0 - 1e-5 and pe.max() <= 1.0 + 1e-5, 'PE values must be in [-1, 1]'
""",
        },
    ],
    "solution": '''def sinusoidal_pe(seq_len, d_model):
    pos = torch.arange(seq_len).float().unsqueeze(1)
    dim = torch.arange(0, d_model, 2).float()
    freqs = 1.0 / (10000.0 ** (dim / d_model))
    angles = pos * freqs
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe''',
    "demo": """pe = sinusoidal_pe(10, 16)
print(pe.shape)
print(pe[:3, :4])""",

}
