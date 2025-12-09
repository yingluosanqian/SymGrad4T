"""
Qwen2 风格的 SwiGLU MLP 符号反向传播示例：
gate = SiLU(x @ w1 + b1)
up   = x @ w3 + b3
y    = (gate * up) @ w2 + b2
loss = CrossEntropy(logits=y, target)

打印 LaTeX，可粘贴到 https://www.latexlive.com/ 预览。
"""

import symgrad4t as sg
from sympy import Symbol


def main():
    batch, in_dim, hidden, up_dim, num_classes = 2, 4, 5, 5, 3

    x = sg.Tensor("x", batch, in_dim)
    w1 = sg.Tensor("w1", in_dim, hidden)
    b1 = sg.Tensor("b1", 1, hidden)
    w3 = sg.Tensor("w3", in_dim, up_dim)
    b3 = sg.Tensor("b3", 1, up_dim)
    w2 = sg.Tensor("w2", up_dim, num_classes)
    b2 = sg.Tensor("b2", 1, num_classes)
    target = sg.Tensor("target", batch, num_classes)

    gate = sg.silu(sg.linear(x, w1, b1, name="gatePre"), name="gate")
    up = sg.linear(x, w3, b3, name="up")
    logits = sg.linear(gate * up, w2, b2, name="logits")
    probs = sg.softmax(logits, name="probs")
    loss = sg.cross_entropy(probs, target, dim=1, name="loss")

    loss._grads.append(Symbol("G"))
    loss.backward()

    block = f"""\\begin{{align*}}
\\text{{Loss expr}} &= {loss.latex_expr()}\\\\
\\text{{Grad w.r.t. }} w1 &= {w1.latex_grad()}\\\\
\\text{{Grad w.r.t. }} w2 &= {w2.latex_grad()}\\\\
\\text{{Grad w.r.t. }} w3 &= {w3.latex_grad()}
\\end{{align*}}"""

    print(block)
    print("将输出粘贴到 https://www.latexlive.com/ 查看公式效果。")


if __name__ == "__main__":
    main()
