"""
Second-order dynamics demo:
    M(q) qdd + C(q,qdot) qdot + c M(q) qdot + grad psi(q) = B(q) qdot
with M(q) = R diag(1, lam_n(d)) R^T, lam_n(d)=1+(sigma/d)^p (p>1),
     B(q) = b(d) J, b(d)=kB * d^alpha.

Generates figures:
  figs2/figS1_trajectories_second_order.png
  figs2/figS2_energy_decay.png
  figs2/figS3_min_distance.png
  figs2/figS4_first_vs_second.png
  figs2/figS5_kmax_safe_second_order.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass
import os

os.makedirs('figs2', exist_ok=True)
J = np.array([[0.0, -1.0],[1.0,  0.0]])

# ---------------------
# Problem data
# ---------------------
@dataclass
class Obstacle:
    c: np.ndarray
    r: float

@dataclass
class Params:
    qg: np.ndarray
    obs: Obstacle
    sigma: float = 0.25
    p: float = 2.0        # metric exponent (>1)
    c_damp: float = 1.5   # Rayleigh damping gain
    kB: float = 1.0       # magnetic gain (two-form magnitude)
    alpha: float = 2.0    # b(d)=kB*d^alpha (use alpha=p for tuning-consistent law)

# ---------------------
# Geometry helpers
# ---------------------
def dist_and_frame(q, obs: Obstacle):
    v = q - obs.c
    Rn = np.linalg.norm(v)
    d = Rn - obs.r
    n = np.array([1.0,0.0]) if Rn < 1e-12 else v / Rn
    t = J @ n
    return d, t, n

def lam_n_of_d(d, sigma, p):
    d_clipped = max(d, 1e-6)
    return 1.0 + (sigma/d_clipped)**p

# ---------------------
# Metric and Christoffels
# ---------------------
def M_of_q(q, prm: Params):
    d, t, n = dist_and_frame(q, prm.obs)
    lam_n = lam_n_of_d(d, prm.sigma, prm.p)
    R = np.column_stack([t, n])
    return R @ np.diag([1.0, lam_n]) @ R.T

def partial_M(q, prm: Params, axis=0, h=1e-4):
    e = np.array([1.0, 0.0]) if axis==0 else np.array([0.0, 1.0])
    Mp = M_of_q(q + h*e, prm)
    Mm = M_of_q(q - h*e, prm)
    return (Mp - Mm)/(2*h)

def christoffel_Gamma(q, prm: Params):
    M = M_of_q(q, prm); Minv = np.linalg.inv(M)
    dM = np.stack([partial_M(q, prm, 0), partial_M(q, prm, 1)], axis=0)  # shape (2,2,2)
    Gamma = np.zeros((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for l in range(2):
                    term = dM[j, l, k] + dM[k, l, j] - dM[l, j, k]
                    s += Minv[i, l]*term
                Gamma[i, j, k] = 0.5*s
    return Gamma

def C_times_qdot(q, qdot, prm: Params):
    Gamma = christoffel_Gamma(q, prm)
    a = np.zeros(2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a[i] += Gamma[i, j, k]*qdot[j]*qdot[k]
    return a

# ---------------------
# Two-form B(q) = b(d) J  and potential
# ---------------------
def b_of_d(d, prm: Params):
    d_clipped = max(d, 1e-6)
    return prm.kB * (d_clipped**prm.alpha)

def B_of_q(q, prm: Params):
    d, _, _ = dist_and_frame(q, prm.obs)
    return b_of_d(d, prm) * J

def grad_psi(q, prm: Params):
    return (q - prm.qg)

# ---------------------
# Second-order RHS on state x = [q; v]
# ---------------------
def second_order_rhs(x, prm: Params):
    q = x[:2]; v = x[2:]
    M = M_of_q(q, prm); Minv = np.linalg.inv(M)
    Cv = C_times_qdot(q, v, prm)
    B  = B_of_q(q, prm)
    rhs_acc = B @ v - Cv - prm.c_damp*(M @ v) - grad_psi(q, prm)   # M a = ...
    a = Minv @ rhs_acc
    return np.hstack([v, a])

def rk4_step(f, x, h, prm):
    k1 = f(x, prm)
    k2 = f(x + 0.5*h*k1, prm)
    k3 = f(x + 0.5*h*k2, prm)
    k4 = f(x + h*k3, prm)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def energy(x, prm: Params):
    q = x[:2]; v = x[2:]
    M = M_of_q(q, prm)
    return 0.5*v.T @ (M @ v) + 0.5*np.linalg.norm(q - prm.qg)**2

# ---------------------
# Drivers
# ---------------------
def simulate_2nd_order(prm, law='dp', q0=np.array([-1.6,-1.4]), v0=np.array([0.0,0.0]),
                       h=0.005, tmax=20.0, tol=1e-3):
    prm_local = Params(qg=prm.qg.copy(), obs=prm.obs, sigma=prm.sigma, p=prm.p,
                       c_damp=prm.c_damp, kB=prm.kB, alpha=prm.alpha)
    if   law == 'none':      prm_local.kB = 0.0
    elif law == 'const':     prm_local.alpha = 0.0
    elif law == 'dpminus1':  prm_local.alpha = prm_local.p - 1.0
    elif law == 'dp':        prm_local.alpha = prm_local.p
    else: raise ValueError('unknown law')

    x = np.hstack([q0, v0])
    traj, Ts, Es = [x.copy()], [0.0], [energy(x, prm_local)]
    for k in range(int(tmax/h)):
        d,_,_ = dist_and_frame(x[:2], prm_local.obs)
        if d < 0: break
        x = rk4_step(second_order_rhs, x, h, prm_local)
        traj.append(x.copy()); Ts.append(Ts[-1]+h); Es.append(energy(x, prm_local))
        if np.linalg.norm(x[:2] - prm_local.qg) < tol: break
    return np.array(Ts), np.array(traj), np.array(Es)

# First-order reference for S4
def first_order_v(q, prm: Params):
    d, t, n = dist_and_frame(q, prm.obs)
    lam_n = lam_n_of_d(d, prm.sigma, prm.p)
    R = np.column_stack([t, n])
    Ginv = R @ np.diag([1.0, 1.0/lam_n]) @ R.T
    gnat = Ginv @ (q - prm.qg)
    beta = prm.kB * (max(d,1e-6)**prm.p)   # β(d)=k d^p
    return -gnat + beta * (J @ gnat)

def simulate_first_order(prm: Params, q0, h=0.01, tmax=25.0, tol=1e-3):
    qs=[q0.copy()]
    for k in range(int(tmax/h)):
        q=qs[-1]
        d,_,_=dist_and_frame(q,prm.obs)
        if d<0: break
        v=first_order_v(q,prm)
        qn = q + h*v
        qs.append(qn)
        if np.linalg.norm(qn - prm.qg) < tol: break
    return np.array(qs)

# ---------------------
# Make figures S1–S5
# ---------------------
if __name__ == '__main__':
    prm = Params(qg=np.array([1.2, 1.1]), obs=Obstacle(c=np.array([0.0,0.0]), r=0.5),
                 sigma=0.25, p=2.0, c_damp=1.5, kB=1.0, alpha=2.0)

    laws = ['none','const','dpminus1','dp']
    colors = {'none':'#1f77b4','const':'#d62728','dpminus1':'#9467bd','dp':'#2ca02c'}
    starts = [np.array([-1.6,-1.4]), np.array([-1.4,0.6]), np.array([0.6,-1.5]), np.array([-1.0,1.6])]

    # S1: trajectories (second-order)
    fig, axes = plt.subplots(1, len(laws), figsize=(4.6*len(laws), 4.4))
    for ax, law in zip(axes, laws):
        ax.add_patch(Circle(prm.obs.c, prm.obs.r, color='k', alpha=0.15))
        ax.plot(prm.qg[0], prm.qg[1], 'r*', ms=12)
        for q0 in starts:
            Ts, Xs, Es = simulate_2nd_order(prm, law=law, q0=q0, v0=np.array([0.0,0.0]), h=0.005, tmax=25.0)
            Qs = Xs[:,:2]
            ax.plot(Qs[:,0], Qs[:,1], '-', color=colors[law], lw=2)
            ax.plot(q0[0], q0[1], 'ko', ms=3)
        ax.set_title(f'2nd-order, B-law: {law}')
        ax.set_xlim([-2,2]); ax.set_ylim([-2,2]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig('figs2/figS1_trajectories_second_order.png', dpi=180); plt.close(fig)

    # S2: energy decay
    q0 = np.array([-1.6,-1.4])
    fig, ax = plt.subplots(1,1, figsize=(6.2,4.2))
    for law in laws:
        Ts, Xs, Es = simulate_2nd_order(prm, law=law, q0=q0, v0=np.array([0.2,0.0]), h=0.005, tmax=20.0)
        ax.plot(Ts, Es, '-', color=colors[law], label=law)
    ax.set_title('Second-order: total energy V(t)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('V=0.5 v^T M v + psi')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); fig.savefig('figs2/figS2_energy_decay.png', dpi=180); plt.close(fig)

    # S3: min distance per law
    mins = {}
    for law in laws:
        md_list=[]
        for q0 in starts:
            Ts, Xs, Es = simulate_2nd_order(prm, law=law, q0=q0, v0=np.array([0.0,0.0]), h=0.005, tmax=25.0)
            ds = [dist_and_frame(Q, prm.obs)[0] for Q in Xs[:,:2]]
            md_list.append(np.min(ds))
        mins[law]=md_list
    fig, ax = plt.subplots(1,1, figsize=(6.0,4.0))
    X = np.arange(len(starts)); wd = 0.18
    for i, law in enumerate(laws):
        ax.bar(X + i*wd, mins[law], width=wd, color=colors[law], label=law)
    ax.set_xticks(X + 1.5*wd); ax.set_xticklabels([f'start {i+1}' for i in range(len(starts))])
    ax.axhline(0.0, color='k', lw=1)
    ax.set_ylabel('min d(t) [m]'); ax.set_title('Second-order: min d(t) per law (higher is safer)')
    ax.grid(True, axis='y', alpha=0.3); ax.legend(ncol=2)
    plt.tight_layout(); fig.savefig('figs2/figS3_min_distance.png', dpi=180); plt.close(fig)

    # S4: first- vs second-order overlay
    q0 = np.array([-1.6,-1.4])
    qs_fo = simulate_first_order(prm, q0)
    Ts, Xs, Es = simulate_2nd_order(prm, law='dp', q0=q0, v0=np.array([0.0,0.0]), h=0.005, tmax=25.0)
    qs_so = Xs[:,:2]
    fig, ax = plt.subplots(1,1, figsize=(6.0,6.0))
    ax.add_patch(Circle(prm.obs.c, prm.obs.r, color='k', alpha=0.15))
    ax.plot(prm.qg[0], prm.qg[1], 'r*', ms=12)
    ax.plot(qs_fo[:,0], qs_fo[:,1], 'b-', lw=2, label='1st-order (β~d^p)')
    ax.plot(qs_so[:,0], qs_so[:,1], 'g--', lw=2, label='2nd-order (B~d^p)')
    ax.plot(q0[0], q0[1], 'ko', ms=4)
    ax.set_aspect('equal'); ax.set_xlim([-2,2]); ax.set_ylim([-2,2])
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('First- vs Second-order trajectory (same tuning law)')
    plt.tight_layout(); fig.savefig('figs2/figS4_first_vs_second.png', dpi=180); plt.close(fig)

    # S5: second-order safe-k estimate (same ring-based formula)
    Ds = np.geomspace(0.01, 0.15, 20)
    Thetas = np.linspace(0, 2*np.pi, 360, endpoint=False)
    kmax_d=[]
    for d in Ds:
        ks=[]
        for th in Thetas:
            q = prm.obs.c + (prm.obs.r + d) * np.array([np.cos(th), np.sin(th)])
            dd, t, n = dist_and_frame(q, prm.obs)
            lam_n = lam_n_of_d(dd, prm.sigma, prm.p)
            Rf = np.column_stack([t, n])
            Ginv = Rf @ np.diag([1.0, 1.0/lam_n]) @ Rf.T
            gnat = Ginv @ (q - prm.qg)
            g_n = float(np.dot(gnat, n)); g_t = abs(float(np.dot(gnat, t)))
            if g_t>1e-12: ks.append(g_n / ( (d**prm.p) * g_t ))
        if ks: kmax_d.append(np.max([k for k in ks if k>0]))
        else:  kmax_d.append(np.nan)
    kmax_d = np.array(kmax_d); k_safe = 0.5*np.nanmin(kmax_d)
    fig, ax = plt.subplots(1,1, figsize=(6.2,4.2))
    ax.plot(Ds, kmax_d, 'o-')
    ax.axhline(k_safe, color='r', ls='--', label=f'suggested k_safe≈{k_safe:.3f}')
    ax.set_xscale('log'); ax.grid(True, which='both', alpha=0.3); ax.legend()
    ax.set_xlabel('distance d'); ax.set_ylabel('k_max(d)')
    ax.set_title('Second-order: conservative safe k (β~d^p / B~d^p)')
    plt.tight_layout(); fig.savefig('figs2/figS5_kmax_safe_second_order.png', dpi=180); plt.close(fig)

    print('Generated second-order figures: figS1..figS5')