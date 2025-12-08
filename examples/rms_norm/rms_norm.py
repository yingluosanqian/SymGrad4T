
import symgrad4t as sg
from sympy import Symbol

# 简单 RMSNorm 示例：y = x / sqrt(mean(x^2)), 然后再与 w 做逐元素乘法。
# 使用符号梯度（Sympy 表达式），最后打印 x 与 w 的梯度（LaTeX 形式）。

# 输入张量形状
batch = 2
hidden_dim = 3

# 定义符号张量
x = sg.Tensor("x", batch, hidden_dim)
w = sg.Tensor("w", batch, hidden_dim)

# 计算 x^2
x2 = sg.Power(x, 2, "x2")

# 按最后一维求均值（先 sum 再 keepdim=True 方便后续广播）
sum_x2 = sg.Sum(x2, dim=-1, keepdim=True, name="sum_sq")
# sqrt(mean)，由于 sum 已 keepdim，除以 hidden_dim 可省略；此处示例直接用 sqrt(sum)
rms = sg.Sqrt(sum_x2, "rms")

# y = x / rms
y = sg.Div(x, rms, "y")
# z = y * w
z = sg.Mul(y, w, "z")

# 反向传播：对 z 注入一个符号梯度 G
z._grads.append(Symbol("G"))
z.backward()

# 打印 x 与 w 的梯度（LaTeX 字符串）
print(x.latex_grad())
print(w.latex_grad())
print("建议将输出的 LaTeX 粘贴到 https://www.latexlive.com/ 查看公式效果")
