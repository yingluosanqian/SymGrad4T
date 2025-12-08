
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
import sympy as sp
from sympy import Symbol
from tensor import Tensor


def test_tensor_shape():
    x = Tensor("x", 1, 2, 3)
    assert x.shape == (1, 2, 3)


def test_tensor_add():
    x = Tensor("x", 1, 2, 3)
    y = Tensor("y", 1, 2, 3)
    z = x + y
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    assert x._grad == g
    assert y._grad == g


def test_tensor_sub():
    x = Tensor("x", 2, 2)
    y = Tensor("y", 2, 2)
    z = x - y
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    assert x._grad == g
    assert y._grad == -g


def test_tensor_mul():
    x = Tensor("x", 2)
    y = Tensor("y", 2)
    z = x * y
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    assert x._grad == g * y.expr
    assert y._grad == g * x.expr


def test_tensor_div():
    x = Tensor("x", 2, 1)
    y = Tensor("y", 2, 1)
    z = x / y
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    assert x._grad == g / y.expr
    assert y._grad == -(g * x.expr) / (y.expr ** 2)


def test_tensor_power():
    x = Tensor("x", 2, 2)
    y = Tensor("y", 2, 2)
    z = x ** y
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    expected_x_grad = g * y.expr * x.expr ** (y.expr - 1)
    expected_y_grad = g * (x.expr ** y.expr) * sp.log(x.expr)
    assert sp.simplify(x._grad - expected_x_grad) == 0
    assert sp.simplify(y._grad - expected_y_grad) == 0


def test_tensor_sqrt():
    x = Tensor("x", 3, 3)
    z = x.sqrt()
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    expected_x_grad = g / (2 * sp.sqrt(x.expr))
    assert sp.simplify(x._grad - expected_x_grad) == 0


def test_tensor_sum():
    x = Tensor("x", 2, 3, 4)
    z = x.sum(1)
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z.shape == (2, 4)
    assert z._grad == g
    assert x._grad == g


def test_tensor_sum_negative_dim():
    x = Tensor("x", 2, 3, 4)
    z = x.sum(-1)
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z.shape == (2, 3)
    assert z._grad == g
    assert x._grad == g


def test_tensor_sum_keepdim():
    x = Tensor("x", 2, 3, 4)
    z = x.sum(1, keepdim=True)
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z.shape == (2, 1, 4)
    assert z._grad == g
    assert x._grad == g


def test_tensor_power_scalar():
    x = Tensor("x", 3, 3)
    z = x ** 3
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    expected = g * 3 * (x.expr ** 2)
    assert sp.simplify(x._grad - expected) == 0


def test_broadcast_add():
    x = Tensor("x", 2, 1)
    y = Tensor("y", 1, 3)
    z = x + y
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z.shape == (2, 3)
    assert z._grad == g
    assert str(x._grad) == "SumDim(G, 1, 1)"
    assert str(y._grad) == "SumDim(G, 0, 1)"


def test_add_scalar_symbol():
    x = Tensor("x", 2)
    a = Tensor("a")
    z = x + a
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z.shape == (2,)
    assert z._grad == g
    assert x._grad == g
    # scalar grad sums over broadcasted axis
    assert str(a._grad) == "SumDim(G, 0, 1)"


def test_latex_output():
    x = Tensor("x", 2, 2)
    y = Tensor("y", 2, 2)
    z = x * y + x
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    # latex of expr should match sympy's rendering
    assert z.latex_expr() == sp.latex(z.expr)
    # latex of grad
    assert z.latex_grad() == sp.latex(g)
