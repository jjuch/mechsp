# poc_no_barrier_metric_magnetic_beta_scaling.py
# Proof-of-concept for invariance-friendly magnetic scaling with NO barrier in psi.
# Generates:
#   figA_invariance_rings.png
#   figB_trajectories_modes.png
#   figC_ring_speeds.png
#   figD_curvature_dp.png
#   figE_corridor_turn.png

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass

SAVE = False # global value; save figures or show them
BREAK = False # Break trajectory simulations if close enough
np.random.seed(1)

@dataclass
class Obstacle:
    c: np.ndarray
    r: float

@dataclass
class Scenario:
    qg: np.ndarray
    obs: Obstacle
    mu: float = 0.0       # << NO barrier in psi
    sigma: float = 0.25   # metric scale
    p: float = 2.0        # exponent for normal amplification (>1)
    beta_k: float = 1.5   # base gain k

# workspace
XMIN, XMAX, YMIN, YMAX = -2.0, 2.0, -2.0, 2.0
J = np.array([[0.0, -1.0],[1.0, 0.0]])

sc = Scenario(qg=np.array([1.2, 1.1]), obs=Obstacle(c=np.array([0.0,0.0]), r=0.5))

# Geometry helpers
def dist_to_obstacle(q, obs):
    return np.linalg.norm(q-obs.c) - obs.r

def unit_normal_to_obstacle(q, obs):
    v = q - obs.c
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return np.array([1.0,0.0])
    return v / nrm

# Potential (NO explicit barrier)
def potential_grad(q, sc: Scenario):
    return (q - sc.qg)  # gradient of 0.5||q-qg||^2

# Metric
def metric_inv(q, sc: Scenario):
    n = unit_normal_to_obstacle(q, sc.obs)
    t = J @ n
    d = max(dist_to_obstacle(q, sc.obs), 1e-6)
    lam_n = 1.0 + (sc.sigma/d)**sc.p
    lam_t = 1.0
    R = np.column_stack([t, n])
    Ginv = R @ np.diag([1.0/lam_t, 1.0/lam_n]) @ R.T
    return Ginv, n, t, d

# Magnetic gain modes
def beta_const(d, sc):
    return sc.beta_k
def beta_d_pminus1(d, sc):
    return sc.beta_k * (d**(sc.p - 1.0))
def beta_d_p(d, sc):
    return sc.beta_k * (d**(sc.p))

# Vector fields
def v_metric(q, sc):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    return -(Ginv @ g)

def v_metric_mag_full(q, sc, beta_fun):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    gnat = Ginv @ g
    beta = beta_fun(d, sc)
    return -(gnat - beta * (J @ gnat))

def v_metric_mag_tan(q, sc, beta_fun):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    gnat = Ginv @ g
    beta = beta_fun(d, sc)
    gnat_n = np.dot(gnat, n)
    m = beta * abs(gnat_n) * t  # tangential-only magnetic term scaled by normal part
    return -(gnat) + m

# Integrator
def rk4_step(q, f, h):
    k1=f(q); k2=f(q+0.5*h*k1); k3=f(q+0.5*h*k2); k4=f(q+h*k3)
    return q + (h/6.0)*(k1+2*k2+2*k3+k4)

def simulate(q0, vf, h=0.01, tmax=30.0, tol=1e-3):
    qs=[q0.copy()]; ts=[0.0]; L=0.0
    for k in range(int(tmax/h)):
        q=qs[-1]
        if dist_to_obstacle(q, sc.obs) < 1e-4:  # collision
            break
        v=vf(q)
        qn = rk4_step(q, vf, h)
        L += np.linalg.norm(qn-q)
        qs.append(qn); ts.append(ts[-1]+h)
        if np.linalg.norm(qn - sc.qg) < tol and BREAK:
            break
    return np.array(ts), np.array(qs), L

# ===== Fig A: Invariance check via ring sampling =====
Ds = np.geomspace(1e-3, 0.3, 36)
Thetas = np.linspace(0, 2*np.pi, 180, endpoint=False)

modes = {
    'metric only': lambda q: v_metric(q, sc),
    'metric+mag (const beta)': lambda q: v_metric_mag_full(q, sc, beta_const),
    'metric+mag (beta ~ d^{p-1})': lambda q: v_metric_mag_full(q, sc, beta_d_pminus1),
    'metric+mag (beta ~ d^{p})': lambda q: v_metric_mag_full(q, sc, beta_d_p),
    'metric+mag TAN (beta ~ d^{p})': lambda q: v_metric_mag_tan(q, sc, beta_d_p),
}

violations = {k: [] for k in modes}
min_n_dot_v = {k: [] for k in modes}
avg_n_dot_v = {k: [] for k in modes}

for d in Ds:
    for name, vf in modes.items():
        nv_vals=[]
        for th in Thetas:
            q = sc.obs.c + (sc.obs.r + d) * np.array([np.cos(th), np.sin(th)])
            n = unit_normal_to_obstacle(q, sc.obs)
            v = vf(q)
            nv_vals.append(np.dot(n, v))
        nv_vals=np.array(nv_vals)
        violations[name].append(np.sum(nv_vals < 0))
        min_n_dot_v[name].append(np.min(nv_vals))
        avg_n_dot_v[name].append(np.mean(nv_vals))

fig, axes = plt.subplots(1, 2, figsize=(12,4))
for name in modes:
    axes[0].plot(Ds, np.array(violations[name])/len(Thetas), label=name)
axes[0].set_xscale('log'); axes[0].set_xlabel('distance d'); axes[0].set_ylabel('fraction of boundary points with n·v < 0')
axes[0].set_title('Nagumo-like invariance violations vs d')
axes[0].grid(True, which='both', alpha=0.3); axes[0].legend(fontsize=8)

for name in modes:
    axes[1].plot(Ds, min_n_dot_v[name], label=name)
axes[1].set_xscale('log'); axes[1].set_xlabel('distance d'); axes[1].set_ylabel('min n·v on ring')
axes[1].set_title('Worst-case boundary pointing (higher ≥ 0 is safer)')
axes[1].grid(True, which='both', alpha=0.3); axes[1].legend(fontsize=8)
plt.tight_layout()
if SAVE:
    fig.savefig('figA_invariance_rings.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# ===== Fig B: Trajectories =====
inits = [np.array([-1.6, -1.4]), np.array([-1.4, 0.6]), np.array([0.6, -1.5]), np.array([-1.0, 1.6])]
sel_modes = [
    ('metric only', modes['metric only']),
    ('metric+mag (const beta)', modes['metric+mag (const beta)']),
    ('metric+mag (beta ~ d^{p-1})', modes['metric+mag (beta ~ d^{p-1})']),
    ('metric+mag (beta ~ d^{p})', modes['metric+mag (beta ~ d^{p})']),
    ('metric+mag TAN (beta ~ d^{p})', modes['metric+mag TAN (beta ~ d^{p})']),
]

fig, axes = plt.subplots(1, 5, figsize=(22,4))
for ax, (name, vf) in zip(axes, sel_modes):
    Nx, Ny = 50, 50
    xs = np.linspace(XMIN, XMAX, Nx)
    ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    U = np.zeros_like(XX); V = np.zeros_like(YY)
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j,i], YY[j,i]])
            if dist_to_obstacle(q, sc.obs) <= 0:
                continue
            vv = vf(q)
            U[j,i], V[j,i] = vv
    sp = np.sqrt(U**2+V**2)
    ax.streamplot(XX,YY,U,V, color=np.clip(sp,0,3), density=1.1, linewidth=0.6, cmap='viridis')
    ax.add_patch(Circle(sc.obs.c, sc.obs.r, color='k', alpha=0.15))
    ax.plot(sc.qg[0], sc.qg[1], 'r*', ms=12)
    for q0 in inits:
        ts, qs, L = simulate(q0, lambda x: vf(x))
        ax.plot(qs[:,0], qs[:,1], '-', lw=2)
        ax.plot(qs[0,0], qs[0,1], 'ko', ms=4)
    ax.set_title(name, fontsize=10)
    ax.set_xlim([XMIN, XMAX]); ax.set_ylim([YMIN, YMAX]); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
plt.tight_layout()
if SAVE:
    fig.savefig('figB_trajectories_modes.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# ===== Fig C: Ring-averaged normal/tangential speeds =====
avg_vn = {k: [] for k in modes}
avg_vt = {k: [] for k in modes}
for d in Ds:
    for name, vf in modes.items():
        vn_vals=[]; vt_vals=[]
        for th in Thetas:
            q = sc.obs.c + (sc.obs.r + d) * np.array([np.cos(th), np.sin(th)])
            n = unit_normal_to_obstacle(q, sc.obs)
            t = J @ n
            v = vf(q)
            vn_vals.append(np.dot(v,n))
            vt_vals.append(np.dot(v,t))
        avg_vn[name].append(np.mean(vn_vals))
        avg_vt[name].append(np.mean(np.abs(vt_vals)))

fig, axes = plt.subplots(1,2, figsize=(12,4))
for name in modes:
    axes[0].plot(Ds, np.maximum(0, avg_vn[name]), label=name)
axes[0].set_xscale('log'); axes[0].set_title('Ring-averaged outward normal speed ⟨max(0,n·v)⟩')
axes[0].set_xlabel('distance d'); axes[0].set_ylabel('speed'); axes[0].grid(True, which='both', alpha=0.3); axes[0].legend(fontsize=8)

for name in modes:
    axes[1].plot(Ds, avg_vt[name], label=name)
axes[1].set_xscale('log'); axes[1].set_title('Ring-averaged |tangential speed| ⟨|t·v|⟩')
axes[1].set_xlabel('distance d'); axes[1].set_ylabel('speed'); axes[1].grid(True, which='both', alpha=0.3); axes[1].legend(fontsize=8)
plt.tight_layout()
if SAVE:
    fig.savefig('figC_ring_speeds.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# ===== Fig D: Curvature maps (β ~ d^p) =====
def curvature_map(vf):
    Nx, Ny = 80, 80
    xs = np.linspace(XMIN, XMAX, Nx)
    ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    K = np.full_like(XX, np.nan, dtype=float)
    eps = 1e-4
    e1 = np.array([1.0,0.0]); e2=np.array([0.0,1.0])
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j,i], YY[j,i]])
            if dist_to_obstacle(q, sc.obs) <= 0:
                continue
            v = vf(q); sp = np.linalg.norm(v)
            if sp < 1e-10: K[j,i]=0.0; continue
            dv_dx = (vf(q+eps*e1)-vf(q-eps*e1))/(2*eps)
            dv_dy = (vf(q+eps*e2)-vf(q-eps*e2))/(2*eps)
            Jv = np.column_stack([dv_dx, dv_dy])
            a = Jv @ v
            cross = v[0]*a[1]-v[1]*a[0]
            K[j,i] = cross/(sp**3)
    return XX,YY,K

vf_full = lambda q: v_metric_mag_full(q, sc, beta_d_p)
vf_tan  = lambda q: v_metric_mag_tan(q, sc, beta_d_p)

XX,YY,K_full = curvature_map(vf_full)
XX,YY,K_tan  = curvature_map(vf_tan)

fig, axes = plt.subplots(1,2, figsize=(12,5))
im0 = axes[0].imshow(np.abs(K_full), origin='lower', extent=[XMIN,XMAX,YMIN,YMAX], cmap='magma', vmin=0, vmax=np.nanpercentile(np.abs(K_full), 99))
axes[0].add_patch(Circle(sc.obs.c, sc.obs.r, color='w', alpha=0.6)); axes[0].plot(sc.qg[0], sc.qg[1], 'c*', ms=12)
axes[0].set_title('|κ|: metric+mag (β~d^p, full J gnat)'); axes[0].set_aspect('equal')
im1 = axes[1].imshow(np.abs(K_tan), origin='lower', extent=[XMIN,XMAX,YMIN,YMAX], cmap='magma', vmin=0, vmax=np.nanpercentile(np.abs(K_tan), 99))
axes[1].add_patch(Circle(sc.obs.c, sc.obs.r, color='w', alpha=0.6)); axes[1].plot(sc.qg[0], sc.qg[1], 'c*', ms=12)
axes[1].set_title('|κ|: metric+mag TAN (β~d^p)'); axes[1].set_aspect('equal')
plt.tight_layout()
if SAVE:
    fig.savefig('figD_curvature_dp.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# ===== Fig E: Corridor/maze turn demo with two discs =====
obs2 = [Obstacle(c=np.array([0.0, 0.0]), r=0.5), Obstacle(c=np.array([0.6, -0.2]), r=0.45)]

@dataclass
class Scenario2:
    qg: np.ndarray
    obs_list: list
    mu: float = 0.0
    sigma: float = 0.25
    p: float = 2.0
    beta_k: float = 1.0

sc2 = Scenario2(qg=np.array([1.5, 1.3]), obs_list=obs2)

def nearest_obs_data(q, obs_list):
    ds = [np.linalg.norm(q-obs.c)-obs.r for obs in obs_list]
    k = int(np.argmin(ds))
    return obs_list[k], ds[k]

def unit_normal_multi(q, obs_list):
    obs, _ = nearest_obs_data(q, obs_list)
    v = q - obs.c
    nrm = np.linalg.norm(v)
    if nrm<1e-12: return np.array([1.0,0.0])
    return v/nrm

def metric_inv_multi(q, sc2: Scenario2):
    n = unit_normal_multi(q, sc2.obs_list)
    t = J @ n
    obs, d = nearest_obs_data(q, sc2.obs_list)
    d = max(d, 1e-6)
    lam_n = 1.0 + (sc2.sigma/d)**sc2.p
    lam_t = 1.0
    R = np.column_stack([t, n])
    Ginv = R @ np.diag([1/lam_t, 1/lam_n]) @ R.T
    return Ginv, n, t, d

def potential_grad2(q, sc2: Scenario2):
    return (q - sc2.qg)

def beta_d_p2(d, sc2):
    return sc2.beta_k * d**(sc2.p)

def v_metric2(q, sc2):
    g = potential_grad2(q, sc2)
    Ginv, n, t, d = metric_inv_multi(q, sc2)
    return -(Ginv @ g)

def v_metric_mag_tan2(q, sc2):
    g = potential_grad2(q, sc2)
    Ginv, n, t, d = metric_inv_multi(q, sc2)
    gnat = Ginv @ g
    beta = beta_d_p2(d, sc2)
    gnat_n = np.dot(gnat, n)
    return -(gnat) + beta * abs(gnat_n) * t

# simulate through corridor
q0 = np.array([-1.5, -1.2])

def simulate2(q0, vf, sc2, h=0.01, tmax=30.0):
    qs=[q0.copy()]
    for k in range(int(tmax/h)):
        q=qs[-1]
        if any([(np.linalg.norm(q-obs.c)-obs.r) < 0 for obs in sc2.obs_list]) and BREAK: break
        qn = rk4_step(q, lambda x: vf(x, sc2), h)
        qs.append(qn)
        if np.linalg.norm(qn - sc2.qg) < 1e-3 and BREAK: break
    return np.array(qs)

qs_m = simulate2(q0, v_metric2, sc2)
qs_t = simulate2(q0, v_metric_mag_tan2, sc2)

fig, ax = plt.subplots(1,1, figsize=(6,6))
Nx, Ny = 60, 60
xs = np.linspace(XMIN, XMAX, Nx)
ys = np.linspace(YMIN, YMAX, Ny)
XX, YY = np.meshgrid(xs, ys)
U=np.zeros_like(XX); V=np.zeros_like(YY)
for i in range(Nx):
    for j in range(Ny):
        q = np.array([XX[j,i], YY[j,i]])
        if any([(np.linalg.norm(q-obs.c)-obs.r) <= 0 for obs in sc2.obs_list]):
            continue
        vv = v_metric_mag_tan2(q, sc2)
        U[j,i], V[j,i] = vv
sp = np.sqrt(U**2+V**2)
ax.streamplot(XX,YY,U,V, color=np.clip(sp,0,3), density=1.0, linewidth=0.6, cmap='viridis')
for obs in sc2.obs_list:
    ax.add_patch(Circle(obs.c, obs.r, color='k', alpha=0.15))
ax.plot(sc2.qg[0], sc2.qg[1], 'r*', ms=12)
ax.plot(qs_m[:,0], qs_m[:,1], 'b-', lw=2, label='metric only')
ax.plot(qs_t[:,0], qs_t[:,1], 'm-', lw=2, label='metric+mag TAN (β~d^p)')
ax.plot(q0[0], q0[1], 'ko', ms=5)
ax.legend(); ax.set_aspect('equal'); ax.set_xlim([XMIN,XMAX]); ax.set_ylim([YMIN,YMAX]); ax.grid(True, alpha=0.3)
ax.set_title('Corridor turn: metric vs metric+mag tangential (β~d^p)')
plt.tight_layout()
if SAVE:
    fig.savefig('figE_corridor_turn.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

print('Generated: figA_invariance_rings.png, figB_trajectories_modes.png, figC_ring_speeds.png, figD_curvature_dp.png, figE_corridor_turn.png')