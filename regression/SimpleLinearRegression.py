import numpy as np


def fit_slr(x, y):
    n = len(x)
    up = 0
    down = 0
    x_average = np.mean(x)
    y_average = np.mean(y)

    for i in range(n):
        up += (x[i] - x_average) * (y[i] - y_average)
        down += np.power((x[i] - x_average), 2)

    b1 = up / float(down)
    b0 = y_average - x_average * b1
    return b0, b1


def predict(x, b0, b1):
    return b0 + x * b1


x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]
b0, b1 = fit_slr(x, y)

print(predict(2, b0, b1))
