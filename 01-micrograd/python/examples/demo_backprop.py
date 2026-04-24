from micrograd import Value


# Forward
a = Value(2.0)
b = Value(3.0)
c = a * b
assert c.data == 6.0
print("ok: forward")


# Сhain: L = (a*b + c) * f
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(-2.0)
L = (a * b + c) * f
L.backward()
assert a.grad == 6.0
assert b.grad == -4.0
print("ok: chain backward")


# a + a — must accumulate
a = Value(3.0)
b = a + a
b.backward()
assert a.grad == 2.0
print("ok: a + a")


# a ** 2
a = Value(3.0)
b = a ** 2
b.backward()
assert a.grad == 6.0
print("ok: a ** 2")


# a / b
a = Value(6.0)
b = Value(2.0)
c = a / b
assert c.data == 3.0
print("ok: a / b")


# Scalar coercion: 2 * a + 1
a = Value(3.0)
b = 2 * a + 1
b.backward()
assert b.data == 7.0
assert a.grad == 2.0
print("ok: scalar coercion")


