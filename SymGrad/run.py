import tensor
from tensor import Tensor, Sum, Sqrt
from sympy import Symbol

batch = 2
hidden_dim = 3

x = Tensor("x", batch, hidden_dim)
w = Tensor("w", batch, hidden_dim)
x2 = tensor.Power(x, 2, "x2")
sum_x2 = Sum(x2, dim=-1, keepdim=True, name="sum_sq")
rms = Sqrt(sum_x2, "rms")
y = tensor.Div(x, rms, "y")
z = tensor.Mul(y, w, "z")
z._grads.append(Symbol("G"))
z.backward()
print(x.latex_grad())
print(w.latex_grad())
