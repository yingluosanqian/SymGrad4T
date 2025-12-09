from symgrad4t.tensor import Tensor, Softmax, CrossEntropy
from symgrad4t.symbol import compute as sc


def linear(x: Tensor, w: Tensor, b: Tensor, name: str | None = None):
    out = (x @ w) + b
    if name is not None:
        out.name = name
        out.display_expr = sc.symbol(name)
    return out


def softmax(logits: Tensor, dim: int = -1, name: str | None = None):
    return Softmax(logits, dim=dim, name=name)


def cross_entropy(logits: Tensor, target: Tensor, dim: int = -1, name: str | None = None):
    return CrossEntropy(logits, target, dim=dim, name=name)


def silu(x: Tensor, name: str | None = None):
    out = x.silu()
    if name is not None:
        out.name = name
        out.display_expr = sc.symbol(name)
    return out
