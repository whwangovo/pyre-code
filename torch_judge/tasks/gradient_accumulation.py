"""Gradient Accumulation task."""

TASK = {
    "title": "Gradient Accumulation",
    "title_zh": "梯度累积",
    "difficulty": "Easy",
    "description_en": "Implement gradient accumulation over micro-batches.\n\nGradient accumulation simulates a large batch by accumulating gradients from multiple smaller batches before a single optimizer step.\n\n**Signature:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`\n\n**Parameters:**\n- `model` — nn.Module\n- `optimizer` — torch optimizer\n- `loss_fn` — loss function\n- `micro_batches` — list of (x, y) tuples\n\n**Returns:** total loss as a float\n\n**Constraints:**\n- Scale each micro-batch loss by `1/n` before backward\n- Must match a single full-batch update numerically",
    "description_zh": "实现微批次梯度累积。\n\n梯度累积通过在多个小批次上累积梯度后执行一次优化器步骤，模拟大批次训练。\n\n**签名:** `accumulated_step(model, optimizer, loss_fn, micro_batches) -> float`\n\n**参数:**\n- `model` — nn.Module\n- `optimizer` — torch 优化器\n- `loss_fn` — 损失函数\n- `micro_batches` — (x, y) 元组列表\n\n**返回:** 总损失（浮点数）\n\n**约束:**\n- 每个微批次损失在反向传播前除以 `n`\n- 必须与单次全批次更新数值一致",
    "function_name": "accumulated_step",
    "hint": "1. optimizer.zero_grad() once\n2. for each micro-batch: forward → loss/n → backward\n3. optimizer.step()\n   Dividing by n ensures accumulated grads match a single full-batch update.",
    "hint_zh": "1. optimizer.zero_grad() 一次\n2. 对每个微批次：前向 → loss/n → 反向\n3. optimizer.step()\n   除以 n 确保累积梯度与单次全批次更新一致。",
    "tests": [
        {
            "name": "Matches full batch update",
            "code": "\nimport torch, torch.nn as nn\ntorch.manual_seed(0)\nmodel = nn.Linear(4, 2, bias=False)\nmodel_ref = nn.Linear(4, 2, bias=False)\nmodel_ref.load_state_dict(model.state_dict())\nloss_fn = nn.MSELoss()\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nopt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)\nx1, y1 = torch.randn(2, 4), torch.randn(2, 2)\nx2, y2 = torch.randn(2, 4), torch.randn(2, 2)\n{fn}(model, opt, loss_fn, [(x1, y1), (x2, y2)])\nopt_ref.zero_grad()\nloss_ref = loss_fn(model_ref(torch.cat([x1, x2])), torch.cat([y1, y2]))\nloss_ref.backward()\nopt_ref.step()\nassert torch.allclose(model.weight.data, model_ref.weight.data, atol=1e-5), 'Must match full batch'\n"
        },
        {
            "name": "Returns loss value",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.01)\nloss = {fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert isinstance(loss, float), f'Should return float, got {type(loss)}'\nassert loss > 0, 'Loss should be positive'\n"
        },
        {
            "name": "Parameters actually update",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nw_before = model.weight.data.clone()\n{fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert not torch.equal(model.weight.data, w_before), 'Should change'\n"
        }
    ],
    "solution": '''def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(micro_batches)
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / n
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss''',
    "demo": """model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Accumulated loss:', loss)""",

}