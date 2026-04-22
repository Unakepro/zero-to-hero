from micrograd import Value
from visualize import draw_dot


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
f = Value(-2.0, label='f')

e = a * b
e.label = 'e'

d = e + c
d.label = 'd'

L = d * f
L.label = 'L'


dot = draw_dot(L)
dot.render('images/graph', view=False, cleanup=True)
print("Saved graph.svg")
