import torch
from micrograd import Value


def test_simple():
    """Compare gradients on a simple expression."""
    # Micrograd
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(-2.0)
    L = (a * b + c) * f
    L.backward()
    
    # PyTorch
    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(-3.0, requires_grad=True)
    c_t = torch.tensor(10.0, requires_grad=True)
    f_t = torch.tensor(-2.0, requires_grad=True)
    L_t = (a_t * b_t + c_t) * f_t
    L_t.backward()
    
    # Compare
    assert abs(L.data - L_t.item()) < 1e-6
    assert abs(a.grad - a_t.grad.item()) < 1e-6
    assert abs(b.grad - b_t.grad.item()) < 1e-6
    assert abs(c.grad - c_t.grad.item()) < 1e-6
    assert abs(f.grad - f_t.grad.item()) < 1e-6
    print("ok: simple expression")


def test_complex():
    """More operations: pow, div, sub, exp."""
    # Micrograd
    a = Value(2.0)
    b = Value(3.0)
    L = (a * b + a ** 2) * (b - 1) / a.exp()
    L.backward()
    
    # PyTorch
    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(3.0, requires_grad=True)
    L_t = (a_t * b_t + a_t ** 2) * (b_t - 1) / torch.exp(a_t)
    L_t.backward()
    
    assert abs(L.data - L_t.item()) < 1e-6
    assert abs(a.grad - a_t.grad.item()) < 1e-6
    assert abs(b.grad - b_t.grad.item()) < 1e-6
    print("ok: complex expression")


def test_tanh():
    """tanh + reused variable"""
    # Micrograd
    a = Value(0.5)
    L = (a + a).tanh() * (a ** 2)
    L.backward()
    
    # PyTorch
    a_t = torch.tensor(0.5, requires_grad=True)
    L_t = torch.tanh(a_t + a_t) * (a_t ** 2)
    L_t.backward()
    
    assert abs(L.data - L_t.item()) < 1e-6
    assert abs(a.grad - a_t.grad.item()) < 1e-6
    print("ok: tanh with reused variable")


def test_long_chain():
    """A longer chain to stress topological sort."""
    # Micrograd
    a = Value(1.5)
    b = Value(-0.5)
    c = a * b
    d = c + a
    e = d * b - a
    f = e.tanh() + (e ** 2)
    f.backward()
    
    # PyTorch
    a_t = torch.tensor(1.5, requires_grad=True)
    b_t = torch.tensor(-0.5, requires_grad=True)
    c_t = a_t * b_t
    d_t = c_t + a_t
    e_t = d_t * b_t - a_t
    f_t = torch.tanh(e_t) + (e_t ** 2)
    f_t.backward()
    
    assert abs(f.data - f_t.item()) < 1e-6
    assert abs(a.grad - a_t.grad.item()) < 1e-6
    assert abs(b.grad - b_t.grad.item()) < 1e-6
    print("ok: long chain")


if __name__ == "__main__":
    test_simple()
    test_complex()
    test_tanh()
    test_long_chain()
    print("\nAll matched PyTorch.")