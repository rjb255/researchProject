#%% Show
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


sx, sy = (400, 500)
x = np.linspace(0, 5, sx)
y = np.linspace(0, 5, sy)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z)

rand = np.random.choice(sx * sy, size=200, replace=False, random_state=1)
_y_p = (rand % sx).astype(int)
_x_p = ((rand - _y_p) / sx).astype(int)

x_p = X[(_x_p, _y_p)]
y_p = Y[_x_p, _y_p]
z_p = Z[_x_p, _y_p] + np.random.normal(scale=0.1, size=150)
plt.plot(x_p, y_p, "x")

df = pd.DataFrame(np.transpose([z_p, x_p, y_p]), columns=["z", "x", "y"])
print(df)
df.to_csv("data.csv")
