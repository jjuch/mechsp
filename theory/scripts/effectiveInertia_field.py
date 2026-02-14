import numpy as np
import matplotlib.pyplot as plt

save = False
x = np.linspace(-1,1,200)
y = np.linspace(-1,1,200)
X,Y = np.meshgrid(x,y)

# Example B1 field (Gaussian)
B1 = np.exp(-(X**2+Y**2))

# Hessian components
B1_xx = np.gradient(np.gradient(B1, x, axis=1), x, axis=1)
B1_yy = np.gradient(np.gradient(B1, y, axis=0), y, axis=0)

trace_M = B1_xx + B1_yy

plt.figure(figsize=(6,5))
plt.contourf(X, Y, trace_M, 40, cmap='viridis')
plt.colorbar(label=r"Trace($M{_eff}$) up to scale")
plt.title(r"Effective Inertia Contribution ∝ $D^2B_1$")
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
if save:
    plt.savefig("effectiveInertia_contrib.png")
else:
    plt.show()