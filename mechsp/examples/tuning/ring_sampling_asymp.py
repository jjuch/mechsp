# Robust near-obstacle asymptotics via ring sampling around obstacle
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

SAVE = False # global value; save figures or show them

@dataclass
class Obstacle:
    c: np.ndarray
    r: float

@dataclass
class Scenario:
    qg: np.ndarray
    obs: Obstacle
    mu: float = 0.6
    sigma: float = 0.2
    p: float = 2.0
    beta0: float = 1.0
    beta_mode: str = 'const'
    beta_prop_k: float = 2.0

J = np.array([[0.0, -1.0],[1.0, 0.0]])

sc = Scenario(qg=np.array([1.2,1.0]), obs=Obstacle(c=np.array([0.0,0.0]), r=0.5))

# vector fields copied

def dist_to_obstacle(q, obs):
    return np.linalg.norm(q-obs.c) - obs.r

def unit_normal_to_obstacle(q, obs):
    v = q - obs.c
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return np.array([1.0,0.0])
    return v/nrm

def potential_grad(q, sc):
    d = max(dist_to_obstacle(q, sc.obs), 1e-4)
    n = unit_normal_to_obstacle(q, sc.obs)
    return (q - sc.qg) - (sc.mu/d)*n

def metric_inv(q, sc):
    n = unit_normal_to_obstacle(q, sc.obs)
    t = J @ n
    d = max(dist_to_obstacle(q, sc.obs), 1e-4)
    lam_n = 1 + (sc.sigma/d)**sc.p
    lam_t = 1.0
    R = np.column_stack([t, n])
    return R @ np.diag([1/lam_t, 1/lam_n]) @ R.T

def beta_gain(q, sc):
    d = max(dist_to_obstacle(q, sc.obs), 1e-4)
    return sc.beta0 if sc.beta_mode=='const' else sc.beta_prop_k * d

def v_baseline(q, sc):
    g = potential_grad(q, sc)
    return -g

def v_metric(q, sc):
    g = potential_grad(q, sc)
    return -(metric_inv(q, sc) @ g)

def v_magnetic_only(q, sc):
    g = potential_grad(q, sc)
    b = beta_gain(q, sc)
    return -(g - b*(J @ g))

def v_metric_magnetic(q, sc):
    g = potential_grad(q, sc)
    gnat = metric_inv(q, sc) @ g
    b = beta_gain(q, sc)
    return -(gnat - b*(J @ gnat))

VF = {
    'Euclidean (baseline)': v_baseline,
    'Metric only': v_metric,
    'Magnetic only': v_magnetic_only,
    'Metric + Magnetic': v_metric_magnetic,
}

# Sample rings around obstacle and compute average |v·n|
Ds = np.geomspace(0.01, 0.4, 25)
Thetas = np.linspace(0, 2*np.pi, 90, endpoint=False)

avg_vn = {k: [] for k in VF}
for d in Ds:
    for name, vf in VF.items():
        vals = []
        for th in Thetas:
            q = sc.obs.c + (sc.obs.r + d) * np.array([np.cos(th), np.sin(th)])
            n = unit_normal_to_obstacle(q, sc.obs)
            v = vf(q, sc)
            vals.append(abs(np.dot(v, n)))
        avg_vn[name].append(np.mean(vals))

# Fit slopes and plot
fig, ax = plt.subplots(figsize=(7,5))
for name, vals in avg_vn.items():
    y = np.array(vals)
    x = Ds
    ax.plot(x, y, '-o', ms=3, label=name)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('distance to obstacle d')
ax.set_ylabel('⟨|v_n|⟩ (ring-avg)')
ax.set_title('Near-obstacle asymptotics via ring sampling')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
if SAVE:
    fig.savefig('fig2b_asymptotics_ring_sampling.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# linear fits for reporting
fits = {}
for name, vals in avg_vn.items():
    x = np.log(Ds)
    y = np.log(np.clip(np.array(vals), 1e-12, None))
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    fits[name] = float(a)

print("Fits: ", fits)