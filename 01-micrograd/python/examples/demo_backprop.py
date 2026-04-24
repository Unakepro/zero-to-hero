from micrograd import Value



# Тест 1: forward работает
a = Value(2.0)
b = Value(3.0)
c = a * b
print(c)                  # Value(data=6.0, grad=0.0)

# Тест 2: _backward это функция, не число
print(type(c._backward))  # <class 'function'>

# Тест 3: backward работает
c.grad = 1.0
c._backward()
print(a.grad)             # 3.0
print(b.grad)             # 2.0

# Тест 4: accumulation для a+a
x = Value(3.0)
y = x + x
y.grad = 1.0
y._backward()
print(x.grad)             # 2.0  (если 1.0 — твой += не работает)