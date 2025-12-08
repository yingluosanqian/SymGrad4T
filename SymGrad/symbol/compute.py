
from typing import List
import sympy as sp


def symbol(name: str):
    return sp.Symbol(name)


def accumulate(grads: List[sp.Expr]):
    if not grads:
        return sp.Integer(0)
    # Add and simplify to keep expressions tidy
    return sp.simplify(sp.Add(*grads))


def subtract(lhs: sp.Expr, rhs: sp.Expr):
    return sp.simplify(lhs - rhs)


def multiply(lhs: sp.Expr, rhs: sp.Expr):
    return sp.simplify(lhs * rhs)


def divide(lhs: sp.Expr, rhs: sp.Expr):
    return sp.simplify(lhs / rhs)


def negative(expr: sp.Expr):
    return sp.simplify(-expr)


def simplify(expr: sp.Expr):
    return sp.simplify(expr)


def power(base: sp.Expr, exponent: sp.Expr):
    return sp.simplify(base ** exponent)


def sqrt(expr: sp.Expr):
    return sp.sqrt(expr)


def log(expr: sp.Expr):
    return sp.log(expr)


def reduce_sum(expr: sp.Expr, dim: int, keepdim: bool = False):
    # Symbolic reduce; keep axis info for readability
    return sp.Function("SumDim")(expr, sp.Integer(dim), sp.Integer(int(keepdim)))


def broadcast(expr: sp.Expr, shape):
    # Annotate broadcast target shape for readability
    shape_nodes = [sp.Integer(s) for s in shape]
    return sp.Function("Broadcast")(expr, sp.Tuple(*shape_nodes))


def to_latex(expr: sp.Expr) -> str:
    return sp.latex(expr)


def expr_from_value(value) -> sp.Expr:
    if isinstance(value, sp.Expr):
        return value
    return sp.sympify(value)
