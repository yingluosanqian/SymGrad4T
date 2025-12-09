
import pytest
import sympy as sp
from sympy import Symbol
from symgrad4t import Tensor


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
    assert sp.simplify(y._grad - expected_y_grad.subs({x.expr ** y.expr: z.display_expr})) == 0


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
    assert str(x._grad) == "Σ(dim=1, keepdim=True)[G]"
    assert str(y._grad) == "Σ(dim=0, keepdim=True)[G]"


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
    assert str(a._grad) == "Σ(dim=0, keepdim=True)[G]"


def test_latex_output():
    x = Tensor("x", 2, 2)
    y = Tensor("y", 2, 2)
    z = x * y + x
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    # latex uses composed expression with parentheses for clarity
    assert z.latex_expr() == "\\left(x\\cdot y\\right)+\\left(x\\right)"
    # latex of grad
    assert z.latex_grad() == sp.latex(g)


def test_max_tensor_tensor():
    x = Tensor("x", 2)
    y = Tensor("y", 2)
    z = x.max(y)
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    assert str(x._grad) == "G*Heaviside(x - y, 0.5)"
    assert str(y._grad) == "G*Heaviside(-x + y, 0.5)"


def test_max_tensor_scalar():
    x = Tensor("x", 3)
    z = x.max(0)
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    zero_broadcast = sp.symbols("0_broadcast")
    assert sp.simplify(x._grad.subs({zero_broadcast: 0}) - g * sp.Heaviside(x.expr, 0.5)) == 0


def test_relu():
    x = Tensor("x", 3)
    z = x.relu()
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z._grad == g
    expected = g * sp.Piecewise((0, x.expr <= 0), (1, True))
    assert str(z.expr) == "ReLU(x)"
    assert sp.simplify(x._grad - expected) == 0


def test_matmul():
    a = Tensor("A", 2, 3)
    b = Tensor("B", 3, 4)
    z = a @ b
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    assert z.shape == (2, 4)
    assert str(a._grad) == "(G) @ ((B)^T)"
    assert str(b._grad) == "((A)^T) @ (G)"


def test_sigmoid():
    x = Tensor("x", 3)
    z = x.sigmoid()
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    sig = 1 / (1 + sp.exp(-x.expr))
    assert sp.simplify(z.expr - sig) == 0
    expected_grad = g * sig * (1 - sig)
    assert sp.simplify(x._grad - expected_grad) == 0


def test_silu():
    x = Tensor("x", 3)
    z = x.silu()
    g = Symbol("G")
    z._grads.append(g)
    z.backward()
    sig = 1 / (1 + sp.exp(-x.expr))
    assert sp.simplify(z.expr - x.expr * sig) == 0
    # SiLU backward includes path through the internal sigmoid:
    # main: g*(sig + x*sig*(1-sig)), plus sigmoid branch: g*x*sig*(1-sig)
    expected_grad = g * (sig + 2 * x.expr * sig * (1 - sig))
    assert sp.simplify(x._grad - expected_grad) == 0
