from typing import List
from .symbol import compute as sc


class Tensor:
    _registry = []

    def __init__(
        self,
        name: str,
        *shape,
        inputs: List["Tensor"] = None,
        expr=None,
        display_expr=None,
    ):
        self.name = name
        self.shape = shape
        self._grads = []
        self._grad = None
        self.inputs = inputs
        self.expr = expr if expr is not None else sc.symbol(name)
        self.display_expr = display_expr if display_expr is not None else self.expr
        Tensor._registry.append(self)

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

    def __matmul__(self, rhs):
        return MatMul(self, rhs)

    def sqrt(self):
        return Sqrt(self)

    def sum(self, dim: int, keepdim: bool = False):
        return Sum(self, dim, keepdim)

    def broadcast_to(self, *shape):
        return Broadcast(self, shape)

    def max(self, rhs):
        return Max(self, rhs)

    def relu(self):
        return ReLU(self)

    def sigmoid(self):
        return Sigmoid(self)

    def silu(self):
        return SiLU(self)

    def softmax(self, dim: int = -1, name: str = None):
        return Softmax(self, dim, name=name, display_name=name)

    def softmax(self, dim: int = -1, name: str = None):
        return Softmax(self, dim, name=name, display_name=name)

    def transpose(self):
        return Transpose(self)

    @property
    def T(self):
        return self.transpose()

    def latex_expr(self):
        return sc.to_latex(self.display_expr)

    def latex_grad(self):
        if self._grad is None:
            return None
        subs_map = {t.expr: t.display_expr for t in Tensor._registry if t.display_expr != t.expr}
        pretty_grad = self._grad.subs(subs_map) if subs_map else self._grad
        return sc.to_latex(pretty_grad)


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
        super().__init__(
            name,
            *out_shape,
            inputs=[lhs, rhs],
            expr=sc.simplify(lhs.expr + rhs.expr),
            display_expr=sc.AddOp(lhs.display_expr, rhs.display_expr),
        )
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
        super().__init__(
            name,
            *out_shape,
            inputs=[lhs, rhs],
            expr=sc.subtract(lhs.expr, rhs.expr),
            display_expr=sc.SubOp(lhs.display_expr, rhs.display_expr),
        )
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
        super().__init__(
            name,
            *out_shape,
            inputs=[lhs, rhs],
            expr=sc.multiply(lhs.expr, rhs.expr),
            # For display, show the actual product instead of a synthetic symbol to avoid odd subscripts
            display_expr=sc.multiply(lhs.display_expr, rhs.display_expr),
        )
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        self.lhs._grads.append(sc.multiply(self._grad, self.rhs.display_expr))
        self.rhs._grads.append(sc.multiply(self._grad, self.lhs.display_expr))
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
        super().__init__(
            name,
            *out_shape,
            inputs=[lhs, rhs],
            expr=sc.divide(lhs.expr, rhs.expr),
            display_expr=sc.DivOp(lhs.display_expr, rhs.display_expr),
        )
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        lhs_grad = sc.divide(self._grad, self.rhs.display_expr)
        rhs_grad = sc.negative(
            sc.divide(
                sc.multiply(self._grad, self.lhs.display_expr),
                sc.multiply(self.rhs.display_expr, self.rhs.display_expr),
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
        super().__init__(name, *out_shape, inputs=inputs, expr=expr, display_expr=sc.display_symbol(name))
        self.lhs = lhs
        self.rhs = rhs if rhs_is_tensor else None
        self.rhs_expr = rhs_expr
        self.rhs_is_tensor = rhs_is_tensor

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # d(lhs^rhs)/dlhs = rhs * lhs^(rhs-1)
        lhs_grad = sc.multiply(self._grad, sc.multiply(self.rhs_expr, sc.power(self.lhs.display_expr, sc.subtract(self.rhs_expr, 1))))
        # d(lhs^rhs)/drhs = lhs^rhs * ln(lhs)
        if self.rhs_is_tensor:
            rhs_grad = sc.multiply(self._grad, sc.multiply(self.display_expr, sc.log(self.lhs.display_expr)))
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
        super().__init__(name, *src.shape, inputs=[src], expr=expr, display_expr=sc.display_symbol(name))
        self.src = src

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # d(sqrt(x))/dx = 1 / (2*sqrt(x))
        denom = sc.multiply(2, sc.sqrt(self.src.display_expr))
        src_grad = sc.divide(self._grad, denom)
        self.src._grads.append(src_grad)
        self.src.backward()


class MatMul(Tensor):
    def __init__(self, lhs: Tensor, rhs: Tensor, name: str = None):
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError("MatMul supports 2D tensors only")
        if lhs.shape[1] != rhs.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {lhs.shape} @ {rhs.shape}")
        out_shape = (lhs.shape[0], rhs.shape[1])
        if name is None:
            name = f"{lhs.name}_matmul_{rhs.name}"
        expr = sc.matmul(lhs.expr, rhs.expr)
        super().__init__(name, *out_shape, inputs=[lhs, rhs], expr=expr, display_expr=sc.display_symbol(name))
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        lhs_grad = sc.matmul(self._grad, sc.transpose(self.rhs.display_expr))
        rhs_grad = sc.matmul(sc.transpose(self.lhs.display_expr), self._grad)
        self.lhs._grads.append(lhs_grad)
        self.rhs._grads.append(rhs_grad)
        self.lhs.backward()
        self.rhs.backward()


class Transpose(Tensor):
    def __init__(self, src: Tensor, name: str = None):
        if len(src.shape) < 2:
            raise ValueError("Transpose expects tensor with rank >= 2")
        if name is None:
            name = f"{src.name}_T"
        out_shape = src.shape[:-2] + (src.shape[-1], src.shape[-2])
        expr = sc.transpose(src.expr)
        super().__init__(name, *out_shape, inputs=[src], expr=expr, display_expr=sc.display_symbol(name))
        self.src = src

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        self.src._grads.append(sc.transpose(self._grad))
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
        super().__init__(name, *out_shape, inputs=[src], expr=expr, display_expr=sc.display_symbol(name))
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
        super().__init__(name, *target_shape, inputs=[src], expr=expr, display_expr=sc.display_symbol(name))
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


class ReLU(Tensor):
    def __init__(self, src: Tensor, name: str = None):
        if name is None:
            name = f"{src.name}_relu"
        expr = sc.relu(src.display_expr)
        super().__init__(name, *src.shape, inputs=[src], expr=expr, display_expr=sc.display_symbol(name))
        self.src = src

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        mask = sc.relu_grad(self.src.display_expr)
        self.src._grads.append(sc.multiply(self._grad, mask))
        self.src.backward()


class Sigmoid(Tensor):
    def __init__(self, src: Tensor, name: str = None, display_name: str = None):
        if name is None:
            name = f"{src.name}_sigmoid"
        expr = sc.divide(1, 1 + sc.exp(sc.negative(src.expr)))
        display_expr = (
            sc.symbol(display_name)
            if display_name is not None
            else sc.SigmoidOp(src.display_expr)
        )
        super().__init__(name, *src.shape, inputs=[src], expr=expr, display_expr=display_expr)
        self.src = src
        self.sig_expr = expr

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        grad = sc.multiply(self._grad, sc.multiply(self.sig_expr, sc.subtract(1, self.sig_expr)))
        self.src._grads.append(grad)
        self.src.backward()


class SiLU(Tensor):
    def __init__(self, src: Tensor, name: str = None):
        if name is None:
            name = f"{src.name}_silu"
        sig = Sigmoid(src)
        expr = sc.multiply(src.expr, sig.expr)
        super().__init__(name, *src.shape, inputs=[src, sig], expr=expr, display_expr=sc.symbol(name))
        self.src = src
        self.sig = sig
        self.sig_expr = sig.expr

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # d(x * sig) = sig + x * sig * (1 - sig)
        term = sc.multiply(self.src.display_expr, sc.multiply(self.sig_expr, sc.subtract(1, self.sig_expr)))
        src_grad = sc.multiply(self._grad, self.sig_expr + term)
        # propagate to src and sig
        self.src._grads.append(src_grad)
        self.sig._grads.append(sc.multiply(self._grad, self.src.display_expr))
        self.src.backward()
        self.sig.backward()
class Softmax(Tensor):
    def __init__(self, src: Tensor, dim: int = -1, name: str = None, display_name: str = None):
        rank = len(src.shape)
        dim = dim + rank if dim < 0 else dim
        if dim < 0 or dim >= rank:
            raise ValueError(f"Invalid dim {dim} for shape {src.shape}")
        auto_name = name if name is not None else f"{src.name}_softmax_dim{dim}"
        exp_x = sc.exp(src.display_expr)
        sum_exp = sc.reduce_sum(exp_x, dim, keepdim=True)
        expr = exp_x / sum_exp
        # Prefer explicit display_name; otherwise if a name is provided, use it for display.
        if display_name is not None:
            display_expr = sc.symbol(display_name)
        elif auto_name is not None:
            display_expr = sc.symbol(auto_name)
        else:
            display_expr = sc.SoftmaxOp(src.display_expr, dim)
        super().__init__(auto_name, *src.shape, inputs=[src], expr=expr, display_expr=display_expr)
        self.src = src
        self.dim = dim

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # s * (g - sum(g*s))
        gs = sc.multiply(self._grad, self.expr)
        sum_gs = sc.reduce_sum(gs, self.dim, keepdim=True)
        src_grad = sc.multiply(self.expr, sc.subtract(self._grad, sum_gs))
        self.src._grads.append(src_grad)
        self.src.backward()


class Max(Tensor):
    def __init__(self, lhs, rhs, name: str = None):
        lhs = _as_tensor(lhs)
        rhs = _as_tensor(rhs)
        lhs, rhs, out_shape = _maybe_broadcast(lhs, rhs)
        if name is None:
            name = f"{lhs.name}_max_{rhs.name}"
        expr = sc.maximum(lhs.display_expr, rhs.display_expr)
        super().__init__(name, *out_shape, inputs=[lhs, rhs], expr=expr, display_expr=sc.display_symbol(name))
        self.lhs = lhs
        self.rhs = rhs

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        # masks using Heaviside for symmetrical tie case
        lhs_mask = sc.heaviside(sc.subtract(self.lhs.display_expr, self.rhs.display_expr))
        rhs_mask = sc.heaviside(sc.subtract(self.rhs.display_expr, self.lhs.display_expr))
        self.lhs._grads.append(sc.multiply(self._grad, lhs_mask))
        self.rhs._grads.append(sc.multiply(self._grad, rhs_mask))
        self.lhs.backward()
        self.rhs.backward()


class CrossEntropy(Tensor):
    def __init__(self, logits: Tensor, target: Tensor, dim: int = -1, name: str = None):
        if logits.shape != target.shape:
            raise ValueError(f"logits shape {logits.shape} and target shape {target.shape} must match")
        self.softmax = logits if isinstance(logits, Softmax) else Softmax(logits, dim=dim, name=None)
        rank = len(target.shape)
        dim = dim + rank if dim < 0 else dim
        out_shape = target.shape[:dim] + target.shape[dim + 1:]
        if name is None:
            name = f"{logits.name}_cross_entropy"
        # -sum(target * log(softmax), dim)
        expr = sc.negative(sc.reduce_sum(sc.multiply(target.expr, sc.log(self.softmax.expr)), dim, keepdim=False))
        super().__init__(name, *out_shape, inputs=[self.softmax, target], expr=expr, display_expr=sc.display_symbol(name))
        self.target = target
        self.dim = dim

    def backward(self):
        self._grad = sc.accumulate(self._grads)
        soft = self.softmax.expr
        # grad w.r.t logits: G * (soft - target)  (softmax+CE简化)
        grad_logits = sc.multiply(self._grad, sc.subtract(soft, self.target.expr))
        self.softmax.src._grads.append(grad_logits)
        # grad w.r.t target: -G * log(soft)
        grad_target = sc.negative(sc.multiply(self._grad, sc.log(soft)))
        self.target._grads.append(grad_target)
        self.softmax.src.backward()
        self.target.backward()
