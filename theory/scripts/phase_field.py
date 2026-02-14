import numpy as np
import matplotlib.pyplot as plt

# Example phase field
save = False
x = np.linspace(-1,1,200)
y = np.linspace(-1,1,200)
X,Y = np.meshgrid(x,y)

phi = np.arctan2(Y, X)
phi_x = np.gradient(phi, x, axis=1)
phi_y = np.gradient(phi, y, axis=0)

perp_x = -phi_y
perp_y =  phi_x

plt.figure(figsize=(6,6))
plt.quiver(X, Y, perp_x, perp_y, color='k', alpha=0.6)
plt.title("Gyroscopic Direction Field ∇⊥φ(q)")
plt.xlabel("x"); plt.ylabel("y")
plt.axis('equal')
plt.tight_layout()
if save:
    plt.savefig('gyroscopic_direction_field.png')
else:
    plt.show()