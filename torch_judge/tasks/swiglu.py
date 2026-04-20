"""SwiGLU Activation task."""

TASK = {
    "title": "SwiGLU Activation",
    "title_zh": "SwiGLU 激活函数",
    "difficulty": "Medium",
    "description_en": "Implement the SwiGLU gated activation function.\n\nSwiGLU is used in LLaMA, PaLM, and other modern LLMs as the MLP activation. It gates one linear projection with a Swish-activated version of another.\n\n**Signature:** `swiglu(x, W1, W2, Wgate) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor `(B, d_model)`\n- `W1` — weight matrix `(d_model, d_ff)`\n- `W2` — output projection `(d_ff, d_model)`\n- `Wgate` — gate weight `(d_model, d_ff)`\n\n**Returns:** output tensor `(B, d_model)`\n\n**Formula:** `hidden = (x @ W1) * swish(x @ Wgate)`, output = `hidden @ W2`\n\n**Swish:** `swish(z) = z * sigmoid(z)`",
    "description_zh": "实现 SwiGLU 门控激活函数。\n\nSwiGLU 被 LLaMA、PaLM 等现代大语言模型用作 MLP 激活函数，用一个线性投影的 Swish 激活版本对另一个进行门控。\n\n**签名:** `swiglu(x, W1, W2, Wgate) -> Tensor`\n\n**参数:**\n- `x` — 输入张量 `(B, d_model)`\n- `W1` — 权重矩阵 `(d_model, d_ff)`\n- `W2` — 输出投影 `(d_ff, d_model)`\n- `Wgate` — 门控权重 `(d_model, d_ff)`\n\n**返回:** 输出张量 `(B, d_model)`\n\n**公式:** 隐藏状态 = `(x @ W1) * swish(x @ Wgate)`，输出 = `隐藏状态 @ W2`\n\n**Swish:** `swish(z) = z * sigmoid(z)`",
    "function_name": "swiglu",
    "hint": "1. `gate = (x @ Wgate) * sigmoid(x @ Wgate)`  ← swish\n2. `hidden = (x @ W1) * gate`\n3. `return hidden @ W2`",
    "hint_zh": "1. `gate = (x @ Wgate) * sigmoid(x @ Wgate)`  ← swish\n2. `hidden = (x @ W1) * gate`\n3. `return hidden @ W2`",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
B, d, d_ff = 2, 16, 32
x = torch.randn(B, d)
W1 = torch.randn(d, d_ff)
W2 = torch.randn(d_ff, d)
Wg = torch.randn(d, d_ff)
out = {fn}(x, W1, W2, Wg)
assert out.shape == (B, d), f'Expected ({B}, {d}), got {out.shape}'
""",
        },
        {
            "name": "Gate is element-wise product",
            "code": """
import torch
torch.manual_seed(1)
B, d, d_ff = 1, 8, 16
x = torch.randn(B, d)
W1 = torch.randn(d, d_ff)
W2 = torch.randn(d_ff, d)
Wg = torch.randn(d, d_ff)
out = {fn}(x, W1, W2, Wg)
# Manually compute expected
h1 = x @ W1
hg = x @ Wg
swish_g = hg * torch.sigmoid(hg)
expected = (h1 * swish_g) @ W2
assert torch.allclose(out, expected, atol=1e-5), f'SwiGLU output mismatch'
""",
        },
        {
            "name": "Gradient flows through gate",
            "code": """
import torch
B, d, d_ff = 2, 8, 16
x = torch.randn(B, d, requires_grad=True)
W1 = torch.randn(d, d_ff, requires_grad=True)
W2 = torch.randn(d_ff, d, requires_grad=True)
Wg = torch.randn(d, d_ff, requires_grad=True)
out = {fn}(x, W1, W2, Wg)
out.sum().backward()
assert x.grad is not None, 'No gradient for x'
assert Wg.grad is not None, 'No gradient for Wgate'
""",
        },
    ],
    "solution": '''def swiglu(x, W1, W2, Wgate):
    h = x @ W1
    gate = x @ Wgate
    swish_gate = gate * torch.sigmoid(gate)
    return (h * swish_gate) @ W2''',
    "demo": """torch.manual_seed(0)
B, D, H = 2, 8, 16
x = torch.randn(B, D)
W1 = torch.randn(D, H)
W2 = torch.randn(H, D)
Wgate = torch.randn(D, H)

out = swiglu(x, W1, W2, Wgate)
print("Output shape:", out.shape)

gate = x @ Wgate
swish_gate = gate * torch.sigmoid(gate)
print("Gate (swish) sample values:", swish_gate[0, :4])""",

}
