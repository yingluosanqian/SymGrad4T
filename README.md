SymGrad4T
=========

一个用 Sympy 做符号表达的极简自动求导/张量广播示例。支持基本张量算子（加减乘除、幂、求和、广播、LaTeX 输出等），附带示例脚本与测试。

安装
----
- 直接通过 PyPI 安装：`pip install symgrad4t`
- 或本地开发模式：`pip install -e .`

运行测试
--------
- 项目根目录执行：`pytest -q`
- 如需只跑库内测试：`cd python && pytest -q`

运行示例
--------
- 示例脚本在 `examples/` 下，如：
  - RMSNorm：`python examples/rms_norm/rms_norm.py`
  - 矩阵四则运算：`python examples/matrix_arith.py`
  - 矩阵求和：`python examples/matrix_sum.py`
  - 三层 MLP：`python examples/mlp.py`
- 输出会打印计算图中张量的梯度（LaTeX 形式），可粘贴到 https://www.latexlive.com/ 查看公式效果。

已支持的 OP
-----------
- 加减乘除：`Add` / `Sub` / `Mul` / `Div`（支持广播与标量/符号）
- 幂：`Power`（支持标量或张量指数，广播）
- 平方根：`Sqrt`
- 维度求和：`Sum(dim, keepdim=False)`（支持负索引与 keepdim）
- 广播：`Broadcast` / `Tensor.broadcast_to`
- LaTeX 输出：`Tensor.latex_expr()` / `Tensor.latex_grad()`
- 其他：`MatMul` / `Transpose` / `Max` / `ReLU` / `Softmax` / `CrossEntropy`
