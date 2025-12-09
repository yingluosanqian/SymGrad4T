
from typing import List
import sympy as sp


class MatMulOp(sp.Function):
    @classmethod
    def eval(cls, lhs, rhs):
        return None

    def _sympystr(self, printer):
        lhs, rhs = self.args
        return f"({printer.doprint(lhs)}) @ ({printer.doprint(rhs)})"

    def _latex(self, printer):
        lhs, rhs = self.args
        return f"\\left({printer._print(lhs)}\\right) @ \\left({printer._print(rhs)}\\right)"


class BroadcastOp(sp.Function):
    @classmethod
    def eval(cls, expr, shape_tuple):
        return None

    def _sympystr(self, printer):
        expr, _ = self.args
        return printer.doprint(expr)

    def _latex(self, printer):
        expr, _ = self.args
        return printer._print(expr)


class SumDimOp(sp.Function):
    @classmethod
    def eval(cls, expr, dim, keepdim):
        return None

    def _sympystr(self, printer):
        expr, dim, keepdim = self.args
        kd = bool(keepdim)
        return f"Σ(dim={dim}, keepdim={kd})[{printer.doprint(expr)}]"

    def _latex(self, printer, **kwargs):
        expr, dim, keepdim = self.args
        kd = bool(keepdim)
        superscript = "^{keep}" if kd else ""
        base = f"\\sum_{{dim={dim}}}{superscript} "
        return base + printer._print(expr)


class ReLUOp(sp.Function):
    @classmethod
    def eval(cls, expr):
        return None

    def _sympystr(self, printer):
        (expr,) = self.args
        return f"ReLU({printer.doprint(expr)})"

    def _latex(self, printer, **kwargs):
        (expr,) = self.args
        return f"\\operatorname{{ReLU}}\\left({printer._print(expr)}\\right)"


class AddOp(sp.Function):
    @classmethod
    def eval(cls, lhs, rhs):
        return None

    def _sympystr(self, printer):
        lhs, rhs = self.args
        return f"({printer.doprint(lhs)}) + ({printer.doprint(rhs)})"

    def _latex(self, printer, **kwargs):
        lhs, rhs = self.args
        return f"\\left({printer._print(lhs)}\\right)+\\left({printer._print(rhs)}\\right)"


class SubOp(sp.Function):
    @classmethod
    def eval(cls, lhs, rhs):
        return None

    def _sympystr(self, printer):
        lhs, rhs = self.args
        return f"({printer.doprint(lhs)}) - ({printer.doprint(rhs)})"

    def _latex(self, printer, **kwargs):
        lhs, rhs = self.args
        return f"\\left({printer._print(lhs)}\\right)-\\left({printer._print(rhs)}\\right)"


class DivOp(sp.Function):
    @classmethod
    def eval(cls, lhs, rhs):
        return None

    def _sympystr(self, printer):
        lhs, rhs = self.args
        return f"({printer.doprint(lhs)}) / ({printer.doprint(rhs)})"

    def _latex(self, printer, **kwargs):
        lhs, rhs = self.args
        return f"\\frac{{{printer._print(lhs)}}}{{{printer._print(rhs)}}}"


class SigmoidOp(sp.Function):
    @classmethod
    def eval(cls, expr):
        return None

    def _sympystr(self, printer):
        (expr,) = self.args
        return f"Sigmoid({printer.doprint(expr)})"

    def _latex(self, printer, **kwargs):
        (expr,) = self.args
        # Use Greek sigma for logistic
        return f"\\sigma\\left({printer._print(expr)}\\right)"


class SoftmaxOp(sp.Function):
    @classmethod
    def eval(cls, expr, dim):
        return None

    def _sympystr(self, printer):
        expr, dim = self.args
        return f"Softmax(dim={dim})({printer.doprint(expr)})"

    def _latex(self, printer, **kwargs):
        expr, dim = self.args
        return f"\\operatorname{{Softmax}}_{{dim={dim}}}\\left({printer._print(expr)}\\right)"


class TransposeOp(sp.Function):
    @classmethod
    def eval(cls, expr):
        return None

    def _sympystr(self, printer):
        (expr,) = self.args
        return f"({printer.doprint(expr)})^T"

    def _latex(self, printer):
        (expr,) = self.args
        return f"\\left({printer._print(expr)}\\right)^{{T}}"


def symbol(name: str):
    return sp.Symbol(name)


def accumulate(grads: List[sp.Expr]):
    if not grads:
        return sp.Integer(0)
    return sp.Add(*grads, evaluate=False)


def subtract(lhs: sp.Expr, rhs: sp.Expr):
    return lhs - rhs


def multiply(lhs: sp.Expr, rhs: sp.Expr):
    # avoid aggressive simplify to keep non-commutative terms intact
    return lhs * rhs


def divide(lhs: sp.Expr, rhs: sp.Expr):
    return lhs / rhs


def negative(expr: sp.Expr):
    return -expr


def simplify(expr: sp.Expr):
    return sp.simplify(expr)


def power(base: sp.Expr, exponent: sp.Expr):
    return sp.simplify(base ** exponent)


def sqrt(expr: sp.Expr):
    return sp.sqrt(expr)


def log(expr: sp.Expr):
    return sp.log(expr)


def exp(expr: sp.Expr):
    return sp.exp(expr)


def reduce_sum(expr: sp.Expr, dim: int, keepdim: bool = False):
    return SumDimOp(expr, sp.Integer(dim), sp.Integer(int(keepdim)))


def broadcast(expr: sp.Expr, shape):
    shape_nodes = [sp.Integer(s) for s in shape]
    return BroadcastOp(expr, sp.Tuple(*shape_nodes))


def to_latex(expr: sp.Expr) -> str:
    return sp.latex(expr, mul_symbol="\\cdot ")


def expr_from_value(value) -> sp.Expr:
    if isinstance(value, sp.Expr):
        return value
    return sp.sympify(value)


def matmul(lhs: sp.Expr, rhs: sp.Expr):
    return MatMulOp(lhs, rhs)


def transpose(expr: sp.Expr):
    return TransposeOp(expr)


def maximum(lhs: sp.Expr, rhs: sp.Expr):
    return sp.Max(lhs, rhs)


class DisplaySymbolOp(sp.Function):
    @classmethod
    def eval(cls, name):
        return None

    def _sympystr(self, printer):
        (name,) = self.args
        return str(name)

    def _latex(self, printer, **kwargs):
        (name,) = self.args
        text = str(name).replace("_", "\\_")
        return f"\\mathrm{{{text}}}"


def display_symbol(name: str):
    # Keep display as plain Symbol to avoid parsing errors in downstream ops
    return sp.Symbol(str(name))


def heaviside(x: sp.Expr):
    # Use sympy Heaviside with value=0.5 at 0 to keep symmetry
    return sp.Heaviside(x, 0.5)


def relu(expr: sp.Expr):
    return ReLUOp(expr)


def relu_grad(expr: sp.Expr):
    # Piecewise gradient: 0 when x<=0, 1 otherwise
    return sp.Piecewise((0, expr <= 0), (1, True))
