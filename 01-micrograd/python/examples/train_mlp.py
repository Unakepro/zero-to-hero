from nn import MLP


# Dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


# Model
mlp = MLP(3, [4, 4, 1])
print(f"Model has {len(mlp.parameters())} parameters")


# Training
for k in range(500):

    # Forward
    ypred = [mlp(x) for x in xs]

    # Loss:
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

    # Zero gradients
    for p in mlp.parameters():
        p.grad = 0.0

    # Backward
    loss.backward()

    # Gradient descent
    for p in mlp.parameters():
        p.data -= 0.05 * p.grad

    print(f"step {k:3d}  loss = {loss.data:.6f}")


# Predictions
print("\nFinal predictions:")
for x, yt in zip(xs, ys):
    yp = mlp(x)
    print(f"  input {x}  target {yt:.1f}  pred {yp.data:.4f}")
