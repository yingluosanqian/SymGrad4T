SymGrad4T
=========

一个用 Sympy 做符号表达的极简自动求导/张量广播示例。支持基本张量算子（加减乘除、幂、求和、广播、LaTeX 输出等），附带示例脚本与测试。

运行测试
--------
- 项目根目录执行：`pytest -q`
- 如需只跑库内测试：`cd SymGrad && pytest -q`

运行示例
--------
- 示例脚本：`SymGrad/run.py`
- 运行命令：`python SymGrad/run.py`
- 默认脚本构建一个简单的归一化与乘法组合，并打印 `x`、`w` 的梯度（LaTeX 形式）。

已支持的 OP
-----------
- 加减乘除：`Add` / `Sub` / `Mul` / `Div`（支持广播与标量/符号）
- 幂：`Power`（支持标量或张量指数，广播）
- 平方根：`Sqrt`
- 维度求和：`Sum(dim, keepdim=False)`（支持负索引与 keepdim）
- 广播：`Broadcast` / `Tensor.broadcast_to`
- LaTeX 输出：`Tensor.latex_expr()` / `Tensor.latex_grad()`
