# Add a variant: magnetic term projected to pure tangent and scaled by the normal component only
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.patches import Circle

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
    beta_mode: str = 'prop_d'  # use the curvature-friendly mode here
    beta_prop_k: float = 2.0

J = np.array([[0.0, -1.0],[1.0, 0.0]])

sc = Scenario(qg=np.array([1.2,1.0]), obs=Obstacle(c=np.array([0.0,0.0]), r=0.5))

# helpers

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
    return R @ np.diag([1/lam_t, 1/lam_n]) @ R.T, n, t, d

def beta_gain(d, sc):
    return sc.beta0 if sc.beta_mode=='const' else sc.beta_prop_k * d

# fields

def v_metric_mag_full(q, sc):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    gnat = Ginv @ g
    beta = beta_gain(d, sc)
    return -(gnat - beta * (J @ gnat))

def v_metric_mag_tan(q, sc):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    gnat = Ginv @ g
    beta = beta_gain(d, sc)
    gnat_n = np.dot(gnat, n)
    # tangential-only magnetic term, scaled by |gnat_n|
    m = beta * abs(gnat_n) * t
    return -(gnat) + m

# simulate from a few inits and plot comparison
inits = [np.array([-1.6, -1.4]), np.array([-1.5,0.0]), np.array([-1.0,1.5]), np.array([0.5,-1.5])]

def rk4_step(q, f, h):
    k1=f(q); k2=f(q+0.5*h*k1); k3=f(q+0.5*h*k2); k4=f(q+h*k3)
    return q + (h/6.0)*(k1+2*k2+2*k3+k4)

def simulate(q0, vf, sc, h=0.01, tmax=30.0, tol=1e-2):
    qs=[q0.copy()]
    for k in range(int(tmax/h)):
        q=qs[-1]
        if dist_to_obstacle(q, sc.obs) < 1e-3: break
        qn = rk4_step(q, lambda x: vf(x, sc), h)
        qs.append(qn)
        if np.linalg.norm(qn - sc.qg) < tol:
            break
    qs=np.array(qs)
    L = np.sum(np.linalg.norm(np.diff(qs,axis=0), axis=1))
    T = len(qs)*0.01
    return qs, L, T

fig, ax = plt.subplots(1,2, figsize=(12,5))
for j, vf in enumerate([v_metric_mag_full, v_metric_mag_tan]):
    for q0 in inits:
        qs, L, T = simulate(q0, vf, sc)
        ax[j].plot(qs[:,0], qs[:,1], '-', lw=2)
        ax[j].plot(qs[0,0], qs[0,1], 'ko', ms=4)
    circ = Circle(sc.obs.c, sc.obs.r, color='k', alpha=0.15)
    ax[j].add_patch(circ)
    ax[j].plot(sc.qg[0], sc.qg[1], 'r*', ms=14)
    ax[j].set_aspect('equal'); ax[j].set_xlim([-2,2]); ax[j].set_ylim([-2,2])
    ax[j].grid(True, alpha=0.3)
    ax[j].set_title(['Metric+Mag (full J gnat)','Metric+Mag (tangent-only)'][j])

plt.tight_layout()
if SAVE:
    fig.savefig('fig5_metricmag_full_vs_tanonly.png', dpi=180)
    plt.close(fig)
else:
    plt.show()
