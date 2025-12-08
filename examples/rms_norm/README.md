# RMSNorm 示例

一个极简的 RMSNorm 前向/反向例子，展示 symgrad4t 的符号自动求导与广播。

## 内容
- `rms_norm.py`：构建计算图 `z = (x / sqrt(sum(x^2))) * w`，对输出注入符号梯度 `G` 并反向传播，打印 `x`、`w` 的梯度（LaTeX 形式）。

## 运行
```bash
python examples/rms_norm/rms_norm.py
```

输出会是两行 LaTeX 公式，分别对应 `x` 与 `w` 的梯度。建议将输出粘贴到 https://www.latexlive.com/ 在线查看公式效果。
