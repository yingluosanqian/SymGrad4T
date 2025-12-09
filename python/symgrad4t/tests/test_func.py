import sympy as sp
from sympy import Symbol

from symgrad4t import Tensor, softmax, cross_entropy, linear
from symgrad4t.symbol.compute import SumDimOp


def test_softmax_grad():
    x = Tensor("x", 2, 3)
    y = softmax(x, dim=1)
    g = Symbol("G")
    y._grads.append(g)
    y.backward()
    assert y._grad == g
    # grad formula: s * (g - sum(g*s))
    gs = sp.simplify(g * y.expr)
    sum_gs = SumDimOp(gs, 1, 1)
    expected = sp.simplify(y.expr * (g - sum_gs))
    assert sp.simplify(x._grad - expected) == 0


def test_cross_entropy():
    logits = Tensor("logits", 2, 2)
    target = Tensor("target", 2, 2)
    loss = cross_entropy(logits, target, dim=1)
    g = Symbol("G")
    loss._grads.append(g)
    loss.backward()
    assert loss.shape == (2,)
    assert loss._grad == g
    # target grad should be -g * log(softmax)
    assert sp.simplify(target._grad + g * sp.log(loss.softmax.expr)) == 0


def test_linear_helper():
    x = Tensor("x", 2, 3)
    w = Tensor("w", 3, 4)
    b = Tensor("b", 1, 4)
    y = linear(x, w, b)
    assert y.shape == (2, 4)
