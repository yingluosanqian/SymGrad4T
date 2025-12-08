"""
矩阵求和示例：
S = sum(X, dim=1)，其中 X 形状为 (3, 4)。

对 S 注入符号梯度 G，反向传播并打印 X 的梯度（LaTeX）。
建议将输出粘贴到 https://www.latexlive.com/ 查看公式效果。
"""

import symgrad4t as sg
from sympy import Symbol


def main():
    # 定义 3x4 矩阵张量
    X = sg.Tensor("X", 3, 4)

    # 前向：沿列求和 -> 形状 (3,)
    S = sg.Sum(X, dim=1, name="S")

    # 反向：在输出 S 上注入符号梯度 G
    S._grads.append(Symbol("G"))
    S.backward()

    print("S expr:", S.latex_expr())
    print("dS/dX:", X.latex_grad())
    print("将输出粘贴到 https://www.latexlive.com/ 查看公式效果")


if __name__ == "__main__":
    main()
