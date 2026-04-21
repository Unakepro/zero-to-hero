class Value:
    def __init__(self, data, _children=(), _op='', label=''):

        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), '+')

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), '*')

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
