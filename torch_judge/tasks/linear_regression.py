"""Linear Regression Three Ways task."""

TASK = {
    "title": "Linear Regression",
    "title_zh": "线性回归",
    "difficulty": "Medium",
    "description_en": "Implement linear regression three ways: closed-form, gradient descent, and nn.Linear.\n\nThis task covers the fundamental approaches to fitting a linear model, from the normal equation to autograd-based training.\n\n**Signature:** `LinearRegression()` (class)\n\n**Methods:**\n- `closed_form(X, y) -> (w, b)` — solve via normal equation\n- `gradient_descent(X, y, lr, steps) -> (w, b)` — manual gradient updates\n- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd training loop\n\n**Constraints:**\n- All three methods should converge to similar weights\n- Closed-form should not use autograd\n- X is (N, D), y is (N,), returns w (D,) and b (scalar)",
    "description_zh": "用三种方式实现线性回归：闭式解、梯度下降和 nn.Linear。\n\n本题涵盖拟合线性模型的基本方法，从正规方程到基于 autograd 的训练。\n\n**签名:** `LinearRegression()`（类）\n\n**方法:**\n- `closed_form(X, y) -> (w, b)` — 通过正规方程求解\n- `gradient_descent(X, y, lr, steps) -> (w, b)` — 手动梯度更新\n- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd 训练循环\n\n**约束:**\n- 三种方法应收敛到相近的权重\n- 闭式解不应使用 autograd\n- X 为 (N, D)，y 为 (N,)，返回 w (D,) 和 b（标量）",
    "function_name": "LinearRegression",
    "hint": "closed_form: augment X with ones col → `lstsq(X_aug, y)` → split `w, b`\ngradient_descent: `grad_w = (2/N) * X.T @ (X@w+b - y)`, `w -= lr*grad_w`\nnn_linear: `nn.Linear(D,1)` + `MSELoss` + `optimizer.step()` loop",
    "hint_zh": "closed_form：X 增加全 1 列 → `lstsq(X_aug, y)` → 拆分 `w, b`\ngradient_descent：`grad_w = (2/N) * X.T @ (X@w+b - y)`，`w -= lr*grad_w`\nnn_linear：`nn.Linear(D,1)` + `MSELoss` + `optimizer.step()` 循环",
    "tests": [
        {
            "name": "Closed-form returns correct shapes",
            "code": """
import torch
torch.manual_seed(42)
X = torch.randn(50, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 3.0 + torch.randn(50) * 0.01
model = {fn}()
w, b = model.closed_form(X, y)
assert w.shape == (3,), f'w shape: {w.shape}, expected (3,)'
assert b.shape == (), f'b shape: {b.shape}, expected scalar'
""",
        },
        {
            "name": "Closed-form finds correct weights",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.closed_form(X, y)
assert torch.allclose(w, true_w, atol=1e-4), f'w: {w} vs true: {true_w}'
assert torch.allclose(b, torch.tensor(true_b), atol=1e-4), f'b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "Gradient descent converges",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.gradient_descent(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'GD w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'GD b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "nn.Linear approach works",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.nn_linear(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'nn w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'nn b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "All three methods agree",
            "code": """
import torch
torch.manual_seed(0)
X = torch.randn(200, 2)
true_w = torch.tensor([1.5, -2.0])
y = X @ true_w + 1.0 + torch.randn(200) * 0.1
model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=3000)
w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=3000)
assert torch.allclose(w_cf, w_gd, atol=0.15), f'CF vs GD: max diff {(w_cf - w_gd).abs().max():.4f}'
assert torch.allclose(w_cf, w_nn, atol=0.15), f'CF vs NN: max diff {(w_cf - w_nn).abs().max():.4f}'
assert abs(b_cf.item() - b_gd.item()) < 0.15, f'Bias CF vs GD: {b_cf.item():.4f} vs {b_gd.item():.4f}'
assert abs(b_cf.item() - b_nn.item()) < 0.15, f'Bias CF vs NN: {b_cf.item():.4f} vs {b_nn.item():.4f}'
""",
        },
        {
            "name": "Closed-form uses no autograd",
            "code": """
import torch
X = torch.randn(30, 2)
y = X @ torch.tensor([1.0, 2.0]) + 0.5
model = {fn}()
w, b = model.closed_form(X, y)
assert not w.requires_grad, 'Closed-form w should not require grad'
""",
        },
    ],
    "solution": '''class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        """Normal equation via augmented matrix."""
        N, D = X.shape
        # Augment X with ones column for bias
        X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # (N, D+1)
        # Solve (X^T X) theta = X^T y
        theta = torch.linalg.lstsq(X_aug, y).solution      # (D+1,)
        w = theta[:D]
        b = theta[D]
        return w.detach(), b.detach()

    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor,
                         lr: float = 0.01, steps: int = 1000):
        """Manual gradient computation — no autograd."""
        N, D = X.shape
        w = torch.zeros(D)
        b = torch.tensor(0.0)

        for _ in range(steps):
            pred = X @ w + b          # (N,)
            error = pred - y           # (N,)
            grad_w = (2.0 / N) * (X.T @ error)  # (D,)
            grad_b = (2.0 / N) * error.sum()     # scalar
            w = w - lr * grad_w
            b = b - lr * grad_b

        return w, b

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        """PyTorch nn.Linear with autograd training loop."""
        N, D = X.shape
        layer = nn.Linear(D, 1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(steps):
            optimizer.zero_grad()
            pred = layer(X).squeeze(-1)  # (N,)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        w = layer.weight.data.squeeze(0)  # (D,)
        b = layer.bias.data.squeeze(0)    # scalar ()
        return w, b''',
    "demo": """torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([2.0, -1.0, 0.5])
y = X @ true_w + 3.0

model = LinearRegression()
for name, method in [("Closed-form", model.closed_form),
                      ("Grad Descent", lambda X, y: model.gradient_descent(X, y, lr=0.05, steps=2000)),
                      ("nn.Linear", lambda X, y: model.nn_linear(X, y, lr=0.05, steps=2000))]:
    w, b = method(X, y)
    print(f"{name:13s}  w={w.tolist()}  b={b.item():.4f}")
print(f"{'True':13s}  w={true_w.tolist()}  b=3.0000")""",

}