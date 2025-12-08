"""
矩阵元素级四则运算示例：
Z = ((A + B) * (A - B)) / B

对 Z 注入符号梯度 G，反向传播并打印 A、B 的梯度（LaTeX）。
建议将输出粘贴到 https://www.latexlive.com/ 查看公式效果。
"""

import symgrad4t as sg
from sympy import Symbol


def main():
    # 定义 2x2 矩阵张量
    A = sg.Tensor("A", 2, 2)
    B = sg.Tensor("B", 2, 2)

    # 前向：Z = ((A + B) * (A - B)) / B
    add = sg.Add(A, B, "add")
    sub = sg.Sub(A, B, "sub")
    mul = sg.Mul(add, sub, "mul")
    Z = sg.Div(mul, B, "Z")

    # 反向：在输出 Z 上注入符号梯度 G
    Z._grads.append(Symbol("G"))
    Z.backward()

    print("Z expr:", Z.latex_expr())
    print("dZ/dA:", A.latex_grad())
    print("dZ/dB:", B.latex_grad())
    print("将输出粘贴到 https://www.latexlive.com/ 查看公式效果")


if __name__ == "__main__":
    main()
