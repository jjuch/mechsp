"""
Reproduces figures A, B, C, D, F and writes the elaborate Markdown file:
  - figs/figA_invariance_rings.png
  - figs/figB_trajectories_modes.png
  - figs/figC_ring_speeds.png
  - figs/figD_curvature_dp.png
  - figs/figF_kmax_safe.png
  - Tuning_with_M_and_N.md  (elaborate version, identical to the one shared in chat)
"""

import os, textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass

# -----------------------------
# Global settings
# -----------------------------
J = np.array([[0.0, -1.0],[1.0,  0.0]])
XMIN, XMAX, YMIN, YMAX = -2.0, 2.0, -2.0, 2.0

@dataclass
class Obstacle:
    c: np.ndarray
    r: float

@dataclass
class Scenario:
    qg: np.ndarray
    obs_list: list
    mu: float = 0.0     # keep barrier-free by default
    sigma: float = 0.25 # boundary-layer scale
    p: float = 2.0      # exponent for normal amplification (>1)
    beta_k: float = 1.0 # magnetic base-gain k

# -----------------------------
# Geometry helpers
# -----------------------------
def nearest_obs_data(q: np.ndarray, obs_list):
    ds = [np.linalg.norm(q-obs.c) - obs.r for obs in obs_list]
    k = int(np.argmin(ds))
    return obs_list[k], ds[k]

def unit_normal_multi(q, obs_list):
    obs, _ = nearest_obs_data(q, obs_list)
    v = q - obs.c
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return np.array([1.0, 0.0])
    return v / nrm

# -----------------------------
# Potential and metric
# -----------------------------
def potential_grad(q, sc: Scenario):
    # barrier-free by default; can toggle sc.mu>0 for comparison
    g_goal = (q - sc.qg)
    if sc.mu == 0.0:
        return g_goal
    obs, d = nearest_obs_data(q, sc.obs_list)
    d = max(d, 1e-6)
    n = unit_normal_multi(q, sc.obs_list)
    g_bar = -(sc.mu/d)*n
    return g_goal + g_bar

def metric_inv(q, sc: Scenario):
    n = unit_normal_multi(q, sc.obs_list)
    t = J @ n
    obs, d = nearest_obs_data(q, sc.obs_list)
    d = max(d, 1e-6)
    lam_n = 1.0 + (sc.sigma/d)**sc.p
    lam_t = 1.0
    R = np.column_stack([t, n])
    Ginv = R @ np.diag([1.0/lam_t, 1.0/lam_n]) @ R.T
    return Ginv, n, t, d

# Magnetic schedules
def beta_const(d, sc: Scenario):      return sc.beta_k
def beta_d_pminus1(d, sc: Scenario):  return sc.beta_k*(d**(sc.p - 1.0))
def beta_d_p(d, sc: Scenario):        return sc.beta_k*(d**(sc.p))

# -----------------------------
# Vector fields
# -----------------------------
def v_metric(q, sc: Scenario):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    return -(Ginv @ g)

def v_metric_mag_full(q, sc: Scenario, beta_fun):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    gnat = Ginv @ g
    beta = beta_fun(d, sc)
    return -(gnat - beta*(J @ gnat))

def v_metric_mag_tan(q, sc: Scenario, beta_fun):
    g = potential_grad(q, sc)
    Ginv, n, t, d = metric_inv(q, sc)
    gnat = Ginv @ g
    beta = beta_fun(d, sc)
    gnat_n = float(np.dot(gnat, n))
    return -(gnat) + beta*abs(gnat_n)*t

# -----------------------------
# Integrator
# -----------------------------
def rk4_step(q, f, h):
    k1 = f(q); k2 = f(q+0.5*h*k1)
    k3 = f(q+0.5*h*k2); k4 = f(q+h*k3)
    return q + (h/6.0)*(k1+2*k2+2*k3+k4)

def simulate_path(q0, vf, sc: Scenario, h=0.01, tmax=30.0, tol=1e-3):
    qs=[q0.copy()]
    for k in range(int(tmax/h)):
        q=qs[-1]
        obs, d = nearest_obs_data(q, sc.obs_list)
        if d < 0: break
        qn = rk4_step(q, lambda x: vf(x, sc), h)
        qs.append(qn)
        if np.linalg.norm(qn - sc.qg) < tol: break
    return np.array(qs)

# -----------------------------
# Figures
# -----------------------------
def fig_invariance_rings(sc: Scenario, name_prefix='A', save_dir='figs'):
    Ds = np.geomspace(1e-3, 0.3, 36)
    Thetas = np.linspace(0, 2*np.pi, 240, endpoint=False)
    modes = {
        'metric only':                    lambda q, sc: v_metric(q, sc),
        'metric+mag (const beta)':        lambda q, sc: v_metric_mag_full(q, sc, beta_const),
        'metric+mag (beta ~ d^{p-1})':    lambda q, sc: v_metric_mag_full(q, sc, beta_d_pminus1),
        'metric+mag (beta ~ d^{p})':      lambda q, sc: v_metric_mag_full(q, sc, beta_d_p),
        'metric+mag TAN (beta ~ d^{p})':  lambda q, sc: v_metric_mag_tan(q, sc, beta_d_p),
    }
    violations = {k: [] for k in modes}
    min_n_dot_v = {k: [] for k in modes}
    for d in Ds:
        for name, vf in modes.items():
            nv_vals=[]
            for th in Thetas:
                q = sc.obs_list[0].c + (sc.obs_list[0].r + d)*np.array([np.cos(th), np.sin(th)])
                n = unit_normal_multi(q, sc.obs_list)
                v = vf(q, sc)
                nv_vals.append(float(np.dot(n, v)))
            nv_vals=np.array(nv_vals)
            violations[name].append(np.mean(nv_vals<0))
            min_n_dot_v[name].append(np.min(nv_vals))
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for name in modes:
        axes[0].plot(Ds, violations[name], label=name)
    axes[0].set_xscale('log'); axes[0].set_xlabel('distance d'); axes[0].set_ylabel('fraction with n·v<0')
    axes[0].set_title('Invariance violations vs d'); axes[0].grid(True, which='both', alpha=0.3); axes[0].legend(fontsize=8)
    for name in modes:
        axes[1].plot(Ds, min_n_dot_v[name], label=name)
    axes[1].set_xscale('log'); axes[1].set_xlabel('distance d'); axes[1].set_ylabel('min n·v on ring')
    axes[1].set_title('Worst-case boundary pointing'); axes[1].grid(True, which='both', alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(save_dir, f'fig{name_prefix}_invariance_rings.png')
    fig.savefig(out, dpi=180); plt.close(fig)
    return out

def fig_trajectories(sc: Scenario, inits, name_prefix='B', save_dir='figs'):
    modes = [
        ('metric only',                      lambda q, sc: v_metric(q, sc)),
        ('metric+mag (const beta)',          lambda q, sc: v_metric_mag_full(q, sc, beta_const)),
        ('metric+mag (beta ~ d^{p-1})',      lambda q, sc: v_metric_mag_full(q, sc, beta_d_pminus1)),
        ('metric+mag (beta ~ d^{p})',        lambda q, sc: v_metric_mag_full(q, sc, beta_d_p)),
        ('metric+mag TAN (beta ~ d^{p})',    lambda q, sc: v_metric_mag_tan(q, sc, beta_d_p)),
    ]
    fig, axes = plt.subplots(1, len(modes), figsize=(4.4*len(modes), 4.2))
    if len(modes)==1: axes=[axes]
    for ax, (title, vf) in zip(axes, modes):
        Nx, Ny = 60, 60
        xs = np.linspace(XMIN, XMAX, Nx)
        ys = np.linspace(YMIN, YMAX, Ny)
        XX, YY = np.meshgrid(xs, ys)
        U = np.zeros_like(XX); V = np.zeros_like(YY)
        for i in range(Nx):
            for j in range(Ny):
                q = np.array([XX[j,i], YY[j,i]])
                coll = any([(np.linalg.norm(q-obs.c)-obs.r) <= 0 for obs in sc.obs_list])
                if coll: continue
                vel = vf(q, sc)
                U[j,i], V[j,i] = vel
        sp = np.sqrt(U**2+V**2)
        ax.streamplot(XX,YY,U,V, color=np.clip(sp,0,3), density=1.0, linewidth=0.6, cmap='viridis')
        for obs in sc.obs_list:
            ax.add_patch(Circle(obs.c, obs.r, color='k', alpha=0.15))
        ax.plot(sc.qg[0], sc.qg[1], 'r*', ms=12)
        for q0 in inits:
            qs = simulate_path(q0, vf, sc)
            ax.plot(qs[:,0], qs[:,1], '-', lw=2); ax.plot(q0[0], q0[1], 'ko', ms=4)
        ax.set_title(title, fontsize=10); ax.set_xlim([XMIN,XMAX]); ax.set_ylim([YMIN,YMAX])
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = os.path.join(save_dir, f'fig{name_prefix}_trajectories_modes.png')
    fig.savefig(out, dpi=180); plt.close(fig)
    return out

def fig_ring_speeds(sc: Scenario, name_prefix='C', save_dir='figs'):
    Ds = np.geomspace(1e-3, 0.3, 36)
    Thetas = np.linspace(0, 2*np.pi, 240, endpoint=False)
    modes = {
        'metric only':                    lambda q, sc: v_metric(q, sc),
        'metric+mag (const beta)':        lambda q, sc: v_metric_mag_full(q, sc, beta_const),
        'metric+mag (beta ~ d^{p-1})':    lambda q, sc: v_metric_mag_full(q, sc, beta_d_pminus1),
        'metric+mag (beta ~ d^{p})':      lambda q, sc: v_metric_mag_full(q, sc, beta_d_p),
        'metric+mag TAN (beta ~ d^{p})':  lambda q, sc: v_metric_mag_tan(q, sc, beta_d_p),
    }
    avg_vn = {k: [] for k in modes}
    avg_vt = {k: [] for k in modes}
    for d in Ds:
        for name, vf in modes.items():
            vn=[]; vt=[]
            for th in Thetas:
                q = sc.obs_list[0].c + (sc.obs_list[0].r + d)*np.array([np.cos(th), np.sin(th)])
                n = unit_normal_multi(q, sc.obs_list); t = J @ n
                v = vf(q, sc)
                vn.append(float(np.dot(v, n)))
                vt.append(abs(float(np.dot(v, t))))
            avg_vn[name].append(np.mean(np.maximum(vn, 0)))
            avg_vt[name].append(np.mean(vt))
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for name in modes:
        axes[0].plot(Ds, avg_vn[name], label=name)
    axes[0].set_xscale('log'); axes[0].set_title('⟨max(0, n·v)⟩ on rings')
    axes[0].set_xlabel('distance d'); axes[0].set_ylabel('outward normal speed')
    axes[0].grid(True, which='both', alpha=0.3); axes[0].legend(fontsize=8)
    for name in modes:
        axes[1].plot(Ds, avg_vt[name], label=name)
    axes[1].set_xscale('log'); axes[1].set_title('⟨|t·v|⟩ on rings')
    axes[1].set_xlabel('distance d'); axes[1].set_ylabel('tangential speed')
    axes[1].grid(True, which='both', alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(save_dir, f'fig{name_prefix}_ring_speeds.png')
    fig.savefig(out, dpi=180); plt.close(fig)
    return out

def curvature_map(sc: Scenario, vf):
    Nx, Ny = 90, 90
    xs = np.linspace(XMIN, XMAX, Nx)
    ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    K = np.full_like(XX, np.nan, dtype=float)
    eps = 1e-4; e1 = np.array([1.0,0.0]); e2=np.array([0.0,1.0])
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j,i], YY[j,i]])
            coll = any([(np.linalg.norm(q-obs.c)-obs.r) <= 0 for obs in sc.obs_list])
            if coll: continue
            v = vf(q, sc); sp = np.linalg.norm(v)
            if sp < 1e-10: K[j,i]=0.0; continue
            dv_dx = (vf(q+eps*e1, sc)-vf(q-eps*e1, sc))/(2*eps)
            dv_dy = (vf(q+eps*e2, sc)-vf(q-eps*e2, sc))/(2*eps)
            Jv = np.column_stack([dv_dx, dv_dy])
            a = Jv @ v
            cross = v[0]*a[1]-v[1]*a[0]
            K[j,i] = cross/(sp**3)
    return XX, YY, K

def fig_curvature_dp(sc: Scenario, name_prefix='D', save_dir='figs'):
    vf_full = lambda q, sc: v_metric_mag_full(q, sc, beta_d_p)
    vf_tan  = lambda q, sc: v_metric_mag_tan(q, sc, beta_d_p)
    XX,YY,K_full = curvature_map(sc, vf_full)
    XX,YY,K_tan  = curvature_map(sc, vf_tan)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].imshow(np.abs(K_full), origin='lower', extent=[XMIN,XMAX,YMIN,YMAX],
                   cmap='magma', vmin=0, vmax=np.nanpercentile(np.abs(K_full),99))
    for obs in sc.obs_list: axes[0].add_patch(Circle(obs.c, obs.r, color='w', alpha=0.6))
    axes[0].plot(sc.qg[0], sc.qg[1], 'c*', ms=12)
    axes[0].set_title('|κ|: metric+mag (β~d^p, full)'); axes[0].set_aspect('equal')
    axes[1].imshow(np.abs(K_tan), origin='lower', extent=[XMIN,XMAX,YMIN,YMAX],
                   cmap='magma', vmin=0, vmax=np.nanpercentile(np.abs(K_tan),99))
    for obs in sc.obs_list: axes[1].add_patch(Circle(obs.c, obs.r, color='w', alpha=0.6))
    axes[1].plot(sc.qg[0], sc.qg[1], 'c*', ms=12)
    axes[1].set_title('|κ|: metric+mag TAN (β~d^p)'); axes[1].set_aspect('equal')
    plt.tight_layout()
    out = os.path.join(save_dir, f'fig{name_prefix}_curvature_dp.png')
    fig.savefig(out, dpi=180); plt.close(fig)
    return out

def fig_kmax_safe(sc: Scenario, d_min=0.01, d_max=0.15, name_prefix='F', save_dir='figs'):
    Ds = np.geomspace(d_min, d_max, 20)
    Thetas = np.linspace(0, 2*np.pi, 360, endpoint=False)
    kmax_d = []
    for d in Ds:
        ks=[]
        for th in Thetas:
            q = sc.obs_list[0].c + (sc.obs_list[0].r + d)*np.array([np.cos(th), np.sin(th)])
            Ginv, n, t, dd = metric_inv(q, sc)
            g = potential_grad(q, sc); gnat = Ginv @ g
            g_n = float(np.dot(gnat, n)); g_t = abs(float(np.dot(gnat, t)))
            if g_t > 1e-10: ks.append(g_n/( (d**sc.p) * g_t ))
        if len(ks): kmax_d.append(np.max([k for k in ks if k>0]))
        else:       kmax_d.append(np.nan)
    kmax_d = np.array(kmax_d)
    k_safe = 0.5*np.nanmin(kmax_d)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(Ds, kmax_d, 'o-', label='ring-wise k_max(d)')
    ax.axhline(k_safe, color='r', linestyle='--', label=f'suggested k_safe≈{k_safe:.3f}')
    ax.set_xscale('log'); ax.set_xlabel('distance d'); ax.set_ylabel('k_max(d)')
    ax.grid(True, which='both', alpha=0.3); ax.legend()
    ax.set_title('Conservative safe k for β(d)=k d^p (no barrier)')
    plt.tight_layout()
    out = os.path.join(save_dir, f'fig{name_prefix}_kmax_safe.png')
    fig.savefig(out, dpi=180); plt.close(fig)
    return out, float(k_safe)


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    os.makedirs('figs', exist_ok=True)

    # Base scenario (one disk obstacle at origin)
    sc = Scenario(qg=np.array([1.2, 1.1]),
                  obs_list=[Obstacle(c=np.array([0.0,0.0]), r=0.5)],
                  mu=0.0, sigma=0.25, p=2.0, beta_k=1.0)

    # Generate figures
    figA = fig_invariance_rings(sc, 'A')
    figB = fig_trajectories(sc,
           [np.array([-1.6,-1.4]), np.array([-1.4,0.6]),
            np.array([0.6,-1.5]),  np.array([-1.0,1.6])],
           'B')
    figC = fig_ring_speeds(sc, 'C')
    figD = fig_curvature_dp(sc, 'D')
    figF, k_safe = fig_kmax_safe(sc, 0.01, 0.15, 'F')

    print("Generated figures A, B, C, D, F and wrote Tuning_with_M_and_N.md")