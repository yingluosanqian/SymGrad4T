SymGrad4T
=========

一个用 Sympy 做符号表达的极简自动求导/张量广播示例，当前版本 **v0.2.0**。已支持经典三层 MLP 的符号反向传播（示例见 `examples/mlp.py`），打印出的 LaTeX 可直接粘贴到 https://www.latexlive.com/ 查看。

安装
----
- PyPI：`pip install symgrad4t`
- 更新：`pip install --upgrade symgrad4t`
- 本地开发：`pip install -e .`

运行测试
--------
- 项目根目录：`pytest -q`
- 仅库内测试：`cd python && pytest -q`

运行示例
--------
- `python examples/rms_norm/rms_norm.py`
- `python examples/matrix_arith.py`
- `python examples/matrix_sum.py`
- `python examples/mlp.py`（三层 MLP 符号反向）

MLP 梯度示例
------------
下图为 `examples/mlp.py` 生成的三层 MLP 梯度表达式（截自 LaTeX 渲染）：

![MLP Grad](media/mlp_grad.png)

已支持的 OP / 功能
------------------
- 元素级：`Add` / `Sub` / `Mul` / `Div`
- 幂与根：`Power`、标量幂、`Sqrt`
- 归约：`Sum(dim, keepdim=False)`（支持负维度）
- 广播：`Broadcast` / `Tensor.broadcast_to`（自动应用于元素级算子）
- 矩阵：`MatMul` / `Transpose`
- 非线性：`Max` / `ReLU` / `Softmax`
- 损失：`CrossEntropy`（内置 Softmax+CE 简化）
- 工具：`Tensor.latex_expr()` / `Tensor.latex_grad()`（乘号用 `\cdot` 提高可读性）
- 其他：`MatMul` / `Transpose` / `Max` / `ReLU` / `Softmax` / `CrossEntropy`
