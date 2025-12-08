from typing import List
from symbol import compute as sc


class Tensor:
    def __init__(
        self,
        name: str,
        *shape,
        inputs: List["Tensor"] = None,
        expr=None,
    ):
        self.name = name
        self.shape = shape
        self._grads = []
        self._grad = None
        self.inputs = inputs
        self.expr = expr if expr is not None else sc.symbol(name)

    def backward(self):
        self._grad = sc.accumulate(self._grads)

    def __expr__(self):
        return f"Tensor({self.name})[{self.shape}])"

    def __add__(self, rhs):
        return Add(self, rhs)

    def __sub__(self, rhs):
        return Sub(self, rhs)

    def __mul__(self, rhs):
        return Mul(self, rhs)

    def __truediv__(self, rhs):
        return Div(self, rhs)

    def __pow__(self, rhs):
        return Power(self, rhs)

    def sqrt(self):
        return Sqrt(self)

    def sum(self, dim: int, keepdim: bool = False):
        return Sum(self, dim, keepdim)

    def broadcast_to(self, *shape):
        return Broadcast(self, shape)

    def latex_expr(self):
        return sc.to_latex(self.expr)

    def latex_grad(self):
        if self._grad is None:
            return None
        return sc.to_latex(self._grad)


def _as_tensor(val):
    if isinstance(val, Tensor):
        return val
    expr = sc.expr_from_value(val)
    name = val.name if hasattr(val, "name") else str(val)
    return Tensor(name, inputs=[], expr=expr)


def _broadcast_shape(shape_a, shape_b):
    # Implements PyTorch-style broadcasting; raises on incompatibility
    result = []
    a_rev = list(reversed(shape_a))
    b_rev = list(reversed(shape_b))
    for i in range(max(len(a_rev), len(b_rev))):
        a_dim = a_rev[i] if i < len(a_rev) else 1
        b_dim = b_rev[i] if i < len(b_rev) else 1
        if a_dim == 1:
            result.append(b_dim)
        elif b_dim == 1:
            result.append(a_dim)
        elif a_dim == b_dim:
            result.append(a_dim)
        else:
            raise ValueError(f"Shapes {shape_a} and {shape_b} not broadcastable")
    return tuple(reversed(result))


def _maybe_broadcast(lhs: Tensor, rhs: Tensor):
    if lhs.shape == rhs.shape:
        return lhs, rhs, lhs.shape
    out_shape = _broadcast_shape(lhs.shape, rhs.shape)
    lhs_b = lhs if lhs.shape == out_shape else lhs.broadcast_to(*out_shape)
    rhs_b = rhs if rhs.shape == out_shape else rhs.broadcast_to(*out_shape)
    return lhs_b, rhs_b, out_shape


class Add(Tensor):
    def __init__(
        self,
        lhs,
        rhs,
        name: str = None,
    ):
        lhs = _as_tensor(lhs)
        rhs = _as_tensor(rhs)
        lhs, rhs, out_shape = _maybe_broadcast(lhs, rhs)
        if name == None:
            name = f"{lhs.name}_add_{rhs.name}"
        super().__init__(name, *out_shape,
                         inputs=[lhs, rhs], expr=sc.simplify(lhs.expr + rhs.expr))
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        self.lhs._grads.append(self._grad)
        self.rhs._grads.append(self._grad)
        self.lhs.backward()
        self.rhs.backward()


class Sub(Tensor):
    def __init__(
        self,
        lhs,
        rhs,
        name: str = None,
    ):
        lhs = _as_tensor(lhs)
        rhs = _as_tensor(rhs)
        lhs, rhs, out_shape = _maybe_broadcast(lhs, rhs)
        if name == None:
            name = f"{lhs.name}_sub_{rhs.name}"
        super().__init__(name, *out_shape,
                         inputs=[lhs, rhs], expr=sc.subtract(lhs.expr, rhs.expr))
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        self.lhs._grads.append(self._grad)
        self.rhs._grads.append(sc.negative(self._grad))
        self.lhs.backward()
        self.rhs.backward()


class Mul(Tensor):
    def __init__(
        self,
        lhs,
        rhs,
        name: str = None,
    ):
        lhs = _as_tensor(lhs)
        rhs = _as_tensor(rhs)
        lhs, rhs, out_shape = _maybe_broadcast(lhs, rhs)
        if name == None:
            name = f"{lhs.name}_mul_{rhs.name}"
        super().__init__(name, *out_shape,
                         inputs=[lhs, rhs], expr=sc.multiply(lhs.expr, rhs.expr))
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        self.lhs._grads.append(sc.multiply(self._grad, self.rhs.expr))
        self.rhs._grads.append(sc.multiply(self._grad, self.lhs.expr))
        self.lhs.backward()
        self.rhs.backward()


class Div(Tensor):
    def __init__(
        self,
        lhs,
        rhs,
        name: str = None,
    ):
        lhs = _as_tensor(lhs)
        rhs = _as_tensor(rhs)
        lhs, rhs, out_shape = _maybe_broadcast(lhs, rhs)
        if name == None:
            name = f"{lhs.name}_div_{rhs.name}"
        super().__init__(name, *out_shape,
                         inputs=[lhs, rhs], expr=sc.divide(lhs.expr, rhs.expr))
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        lhs_grad = sc.divide(self._grad, self.rhs.expr)
        rhs_grad = sc.negative(
            sc.divide(
                sc.multiply(self._grad, self.lhs.expr),
                sc.multiply(self.rhs.expr, self.rhs.expr),
            )
        )
        self.lhs._grads.append(lhs_grad)
        self.rhs._grads.append(rhs_grad)
        self.lhs.backward()
        self.rhs.backward()


class Power(Tensor):
    def __init__(
        self,
        lhs: Tensor,
        rhs,
        name: str = None,
    ):
        rhs_is_tensor = isinstance(rhs, Tensor)
        if rhs_is_tensor:
            lhs, rhs, out_shape = _maybe_broadcast(lhs, rhs)
            rhs_expr = rhs.expr
        else:
            out_shape = lhs.shape
            rhs_expr = sc.simplify(rhs)
        if name is None:
            name = f"{lhs.name}_pow_{rhs.name}" if rhs_is_tensor else f"{lhs.name}_pow_const"
        expr = sc.power(lhs.expr, rhs_expr)
        inputs = [lhs, rhs] if rhs_is_tensor else [lhs]
        super().__init__(name, *out_shape, inputs=inputs, expr=expr)
        self.lhs = lhs
        self.rhs = rhs if rhs_is_tensor else None
        self.rhs_expr = rhs_expr
        self.rhs_is_tensor = rhs_is_tensor

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # d(lhs^rhs)/dlhs = rhs * lhs^(rhs-1)
        lhs_grad = sc.multiply(self._grad, sc.multiply(self.rhs_expr, sc.power(self.lhs.expr, sc.subtract(self.rhs_expr, 1))))
        # d(lhs^rhs)/drhs = lhs^rhs * ln(lhs)
        if self.rhs_is_tensor:
            rhs_grad = sc.multiply(self._grad, sc.multiply(self.expr, sc.log(self.lhs.expr)))
        self.lhs._grads.append(lhs_grad)
        if self.rhs_is_tensor:
            self.rhs._grads.append(rhs_grad)
        self.lhs.backward()
        if self.rhs_is_tensor:
            self.rhs.backward()


class Sqrt(Tensor):
    def __init__(self, src: Tensor, name: str = None):
        if name is None:
            name = f"{src.name}_sqrt"
        expr = sc.sqrt(src.expr)
        super().__init__(name, *src.shape, inputs=[src], expr=expr)
        self.src = src

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # d(sqrt(x))/dx = 1 / (2*sqrt(x))
        denom = sc.multiply(2, sc.sqrt(self.src.expr))
        src_grad = sc.divide(self._grad, denom)
        self.src._grads.append(src_grad)
        self.src.backward()


class Sum(Tensor):
    def __init__(self, src: Tensor, dim: int, keepdim: bool = False, name: str = None):
        rank = len(src.shape)
        dim = dim + rank if dim < 0 else dim
        if dim < 0 or dim >= rank:
            raise ValueError(f"Invalid dim {dim} for shape {src.shape}")
        if name is None:
            name = f"{src.name}_sum_dim{dim}"
        if keepdim:
            out_shape = src.shape[:dim] + (1,) + src.shape[dim + 1:]
        else:
            out_shape = src.shape[:dim] + src.shape[dim + 1:]
        expr = sc.reduce_sum(src.expr, dim, keepdim=keepdim)
        super().__init__(name, *out_shape, inputs=[src], expr=expr)
        self.src = src
        self.dim = dim
        self.keepdim = keepdim

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # Gradient broadcasts along the reduced dimension
        self.src._grads.append(self._grad)
        self.src.backward()


class Broadcast(Tensor):
    def __init__(self, src: Tensor, shape, name: str = None):
        target_shape = tuple(shape)
        _ = _broadcast_shape(src.shape, target_shape)
        if name is None:
            name = f"{src.name}_broadcast"
        expr = sc.broadcast(src.expr, target_shape)
        super().__init__(name, *target_shape, inputs=[src], expr=expr)
        self.src = src
        self.target_shape = target_shape

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # Sum gradients along broadcasted axes
        src_shape = self.src.shape
        tgt_shape = self.target_shape
        rank_src = len(src_shape)
        rank_tgt = len(tgt_shape)
        grad = self._grad
        # Reduce extra leading dims
        for axis in range(rank_tgt - rank_src):
            grad = sc.reduce_sum(grad, axis, keepdim=True)
        # Reduce axes where src dim is 1
        for axis in range(rank_tgt - rank_src, rank_tgt):
            src_dim = src_shape[axis - (rank_tgt - rank_src)]
            tgt_dim = tgt_shape[axis]
            if src_dim == 1 and tgt_dim != 1:
                grad = sc.reduce_sum(grad, axis, keepdim=True)
        self.src._grads.append(grad)
        self.src.backward()
