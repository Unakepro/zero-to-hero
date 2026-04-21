from micrograd import Value


# -- build a small graph: L = (a * b + c) * f --
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
f = Value(-2.0, label='f')

e = a * b
e.label = 'e'      # e = a * b = -6.0

d = e + c
d.label = 'd'      # d = e + c =  4.0

L = d * f
L.label = 'L'      # L = d * f = -8.0


# -- Inspect forward pass reusults --
print("Forward pass results:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  c = {c}")
print(f"  f = {f}")
print(f"  e = a * b = {e}")
print(f"  d = e + c = {d}")
print(f"  L = d * f = {L}")


# -- check operators works on both sides --
print("\nOperators work on both sides:")
x = Value(3.0, label='x')
print(f"  x * 2  = {x * 2}")
print(f"  2 * x  = {2 * x}")
print(f"  x + 1  = {x + 1}")
print(f"  1 + x  = {1 + x}")
