"""
三层 MLP 示例：x -> Linear1 -> ReLU -> Linear2 -> Softmax -> CrossEntropy
使用符号梯度，打印 logits/损失的表达式与输入权重的梯度。
"""

import symgrad4t as sg
from sympy import Symbol


def main():
    batch, in_dim, hidden, num_classes = 2, 4, 5, 3

    # 输入与参数
    x = sg.Tensor("x", batch, in_dim)
    w1 = sg.Tensor("w1", in_dim, hidden)
    b1 = sg.Tensor("b1", 1, hidden)
    w2 = sg.Tensor("w2", hidden, num_classes)
    b2 = sg.Tensor("b2", 1, num_classes)
    target = sg.Tensor("target", batch, num_classes)  # one-hot/概率分布

    # 前向
    h1 = sg.ReLU(sg.linear(x, w1, b1, name="h1"), name="relu1")
    logits = sg.linear(h1, w2, b2, name="logits")
    probs = sg.softmax(logits, dim=1, name="probs")
    # 交叉熵直接用 softmax 后的 probs，避免重复 softmax
    loss = sg.cross_entropy(probs, target, dim=1, name="loss")

    # 反向：对 loss 注入符号梯度 G
    loss._grads.append(Symbol("G"))
    loss.backward()

    loss_tex = loss.latex_expr()
    w1_tex = w1.latex_grad()
    w2_tex = w2.latex_grad()

    block = f"""\\begin{{align*}}
\\text{{Loss expr}} &= {loss_tex}\\\\
\\text{{Grad w.r.t. }} w1 &= {w1_tex}\\\\
\\text{{Grad w.r.t. }} w2 &= {w2_tex}
\\end{{align*}}"""

    print("LaTeX block (可直接粘贴到 https://www.latexlive.com/ ):")
    print(block)
    print()
    print("将输出的 LaTeX 粘贴到 https://www.latexlive.com/ 查看公式效果")


if __name__ == "__main__":
    main()
