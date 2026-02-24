import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Geometric Riemannian+Magnetic navigation proof of concept
# ----------------------

# Constants
m0 = 0.015   # kg, base mass
k = 1.0      # potential stiffness
c_damp = 0.07  # viscous damping coefficient (N*s/m)

# Domain and scenario
q0 = np.array([-0.08, 0.0])
qg = np.array([ 0.08, 0.0])
obs_c = np.array([0.0, 0.0])
r = 0.02

# Metric and magnetic shaping parameters (to be tuned)
alpha = 6e-4   # strength for metric inflation near obstacle
beta  = 6e-4   # strength for magnetic swirl near obstacle

# Smoothing / singularity control
EPS = 1e-4

# Time integration
T = 12.0
h = 1e-3
N = int(T/h)

# Helper
J = np.array([[0.0, -1.0], [1.0, 0.0]])

def signed_distance(q):
    return np.linalg.norm(q - obs_c) - r

def shape_profile(d):
    return 1.0 / (d*d + EPS*EPS)

def B_field(q):
    return beta * shape_profile(signed_distance(q))

def M_matrix(q):
    qvec = q - obs_c
    rho = np.linalg.norm(qvec) + 1e-12
    n = qvec / rho
    s = shape_profile(signed_distance(q))
    # Inflate mass in normal direction
    return m0 * np.eye(2) + alpha * s * np.outer(n, n)

def grad_V(q):
    return k * (q - qg)

def dM_numeric(q, hfd=1e-6):
    M0 = M_matrix(q)
    dMx = (M_matrix(q + np.array([hfd, 0.0])) - M0) / hfd
    dMy = (M_matrix(q + np.array([0.0, hfd])) - M0) / hfd
    return dMx, dMy

def christoffel_contraction(q, v):
    # G_i = sum_{j,k} Gamma_{i j k} v_j v_k
    dMx, dMy = dM_numeric(q)
    dM = [dMx, dMy]  # index 0 -> d/dx, 1 -> d/dy
    G = np.zeros(2)
    for i in range(2):
        acc = 0.0
        for j in range(2):
            for k in range(2):
                dM_ik_dqj = dM[j][i, k]
                dM_ij_dqk = dM[k][i, j]
                dM_jk_dqi = dM[i][j, k]
                Gamma_ijk = 0.5 * (dM_ik_dqj + dM_ij_dqk - dM_jk_dqi)
                acc += Gamma_ijk * v[j] * v[k]
        G[i] = acc
    return G

def N_dot_v(q, v):
    B = B_field(q)
    return B * (J @ v)

def f_state(t, y):
    # state y = [x, y, vx, vy]
    q = y[0:2]
    v = y[2:4]
    M = M_matrix(q)
    G = christoffel_contraction(q, v)
    Nv = N_dot_v(q, v)
    F = - G - Nv - grad_V(q) - c_damp * v
    a = np.linalg.solve(M, F)
    return np.hstack((v, a))

def rk4_step(t, y, h):
    k1 = f_state(t, y)
    k2 = f_state(t + 0.5*h, y + 0.5*h*k1)
    k3 = f_state(t + 0.5*h, y + 0.5*h*k2)
    k4 = f_state(t + h,     y + h*k3)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Integrate
traj = np.zeros((N+1, 4))
traj[0, 0:2] = q0
traj[0, 2:4] = np.array([0.0, 0.0])

hit_obstacle = False
reached_goal = False

for n in range(N):
    t = n*h
    y = traj[n]
    # Goal condition: close in position and nearly stopped
    if np.linalg.norm(y[0:2] - qg) < 1e-3 and np.linalg.norm(y[2:4]) < 2e-3:
        traj = traj[:n+1]
        reached_goal = True
        break
    # Obstacle intrusion stop
    if np.linalg.norm(y[0:2] - obs_c) <= r:
        traj = traj[:n+1]
        hit_obstacle = True
        break
    y_next = rk4_step(t, y, h)
    traj[n+1] = y_next

# Plot
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.set_aspect('equal', 'box')
ax.set_xlim(-0.15, 0.15)
ax.set_ylim(-0.15, 0.15)
ax.grid(True, alpha=0.25)
ax.plot(traj[:,0], traj[:,1], 'b-', lw=2, label='trajectory')
ax.plot(q0[0], q0[1], 'go', label='start q0')
ax.plot(qg[0], qg[1], 'rx', ms=8, mew=2, label='goal qg')
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(obs_c[0] + r*np.cos(theta), obs_c[1] + r*np.sin(theta), 'k-', lw=2, label='obstacle')
ax.legend(loc='upper left')
ax.set_title('Riemannian + Magnetic Navigation (M, B, V)')
plt.tight_layout()
plt.show()