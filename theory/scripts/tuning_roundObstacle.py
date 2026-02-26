"""
Second-order model with full Levi–Civita and gyroscopic two-form:

  L = 1/2 v^T M(q) v + A(q)·v - psi(q)
  EOM: M(q) qdd + C(q,qdot) qdot + c M(q) qdot + grad psi = N(q) qdot

Metric (your proposal):
  M(q) = m0 * I_2 + alpha * s(d) * n n^T,
  s(d) = 1/(d^2 + eps^2),  n(q) = (q - c)/||q - c||,  d(q) = ||q - c|| - r.

Gyroscopic two-form (skew):
  N(q) = B(q) J,  test the laws: none, const, d^{p-1}, d^{p}  (final recommendation: d^{p}).

Generates figures (saved under ./figs):
    This can be done optionally for an optimized set of parameters.
    A  invariance violations vs d  &  worst-case boundary pointing
    B  trajectories for: no-metric&no-mag, metric-only, metric+mag (const),metric+mag (d^{p-1}), metric+mag (d^p), metric+mag TAN (d^p)  + background streamlines
    C  ⟨max(0, n·a)⟩ and ⟨|t·a|⟩ on rings evaluated at grazing (second-order analog of speed plots)
    D  curvature maps |κ| for full vs TAN with b(d)~d^p, evaluated at canonical grazing speed
    F  conservative safe k for b(d)=k d^p (ring-based, via natural gradient under M(q))
"""

import os, numpy as np
import sys, json, csv, itertools, random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass

SAVE = False # Save or show figures

os.makedirs('figs', exist_ok=True)
J = np.array([[0.0, -1.0],[1.0,  0.0]])
XMIN, XMAX, YMIN, YMAX = -2.0, 2.0, -2.0, 2.0

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
    m0: float = 0.015 # kg
    alpha: float = 1.2
    eps: float = 0.05
    p: float = 2.0          # tuning exponent (>1)
    c_damp: float = 1.15    # small to show underdamping
    kB: float = 1.1
    k_psi: float = 1.0      # stiffness potential field \psi

def dist_n_t(q, obs: Obstacle):
    v = q - obs.c
    Rn = np.linalg.norm(v)
    d = Rn - obs.r
    n = np.array([1.0,0.0]) if Rn < 1e-12 else v/Rn
    t = J @ n
    return d, n, t

def s_of_d(d, eps):
    # one-sided: only outside is relevant; clamp at 0 to avoid negative blow-up
    return 1.0/((max(d, 0.0))**2 + eps**2)

def M_of_q(q, prm: Params):
    d, n, _ = dist_n_t(q, prm.obs)
    s = s_of_d(d, prm.eps)
    M = prm.m0*np.eye(2) + prm.alpha*s*np.outer(n,n)
    return 0.5*(M+M.T)

# Christoffels by FD (small h, symmetric M)
def partial_M(q, prm: Params, axis=0, h=2e-4):
    e = np.array([1.0, 0.0]) if axis==0 else np.array([0.0, 1.0])
    Mp = M_of_q(q + h*e, prm)
    Mm = M_of_q(q - h*e, prm)
    return (Mp - Mm)/(2*h)

def christoffel(q, prm: Params):
    M = M_of_q(q, prm); Minv = np.linalg.inv(M)
    dMx = partial_M(q, prm, 0); dMy = partial_M(q, prm, 1)
    dM = np.stack([dMx, dMy], axis=0)
    Gamma = np.zeros((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for l in range(2):
                    term = dM[j,l,k] + dM[k,l,j] - dM[l,j,k]
                    s += Minv[i,l]*term
                Gamma[i,j,k] = 0.5*s
    return Gamma

def C_times_qdot(q, v, prm: Params):
    G = christoffel(q, prm)
    a = np.zeros(2)
    for i in range(2):
        a[i] = v @ G[i] @ v
    return a


# Natural gradient pieces under M(q) (for diagnostics & TAN scaling)
def Ginv_of_q(q, prm: Params):
    d, n, t = dist_n_t(q, prm.obs)
    s = s_of_d(d, prm.eps)
    Mt = prm.m0
    Mn = prm.m0 + prm.alpha*s
    R = np.column_stack([t,n])
    return R @ np.diag([1.0/Mt, 1.0/Mn]) @ R.T, n, t, d

def gnat_components(q, prm: Params):
    Ginv, n, t, d = Ginv_of_q(q, prm)
    gnat = Ginv @ (q - prm.qg)
    return float(t @ gnat), float(n @ gnat), n, t, d


# N(q) = B(q) J = B(d(q)) J
def phi_window(d, d_on=0.35, q=2.0):
    # Gaussian-like window ~1 near obstacle, decays after d_on
    return np.exp(-(max(d,0.0)/d_on)**q)

def b_of_d(d, prm: Params, law: str):
    dpos = max(d, 1e-6)
    if   law == 'none':     return 0.0
    elif law == 'const':    return prm.kB
    elif law == 'dpminus1': return prm.kB*(dpos**(prm.p-1.0)) * phi_window(dpos)
    elif law == 'dp':       return prm.kB*(dpos**(prm.p)) * phi_window(dpos)
    else: raise ValueError('unknown law')


def b_of_d_TAN(q, prm: Params):
    # TAN-scaled magnitude: k * d^p * |g_n^natural|
    gt, gn, n, t, d = gnat_components(q, prm)
    return prm.kB*(max(d,1e-6)**prm.p)*abs(gn)*phi_window(d)

def N_of_q(q, prm: Params, law: str):
    if law == 'tan':
        return b_of_d_TAN(q, prm)*J
    d,_,_ = dist_n_t(q, prm.obs)
    return b_of_d(d, prm, law)*J

def grad_psi(q, prm: Params):
    return prm.k_psi*(q - prm.qg)

# second-order RHS on x=[q;v]
def rhs_second_order(x, prm: Params, law: str):
    q = x[:2]; v = x[2:]
    M = M_of_q(q, prm); Minv = np.linalg.inv(M)
    Cv = C_times_qdot(q, v, prm)
    N  = N_of_q(q, prm, law)
    rhs_acc = N @ v - Cv - prm.c_damp*(M @ v) - grad_psi(q, prm)
    a = Minv @ rhs_acc
    return np.hstack([v, a])

def rk4(x, h, f, prm, law):
    k1 = f(x, prm, law)
    k2 = f(x + 0.5*h*k1, prm, law)
    k3 = f(x + 0.5*h*k2, prm, law)
    k4 = f(x + h*k3, prm, law)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def energy(x, prm: Params):
    q = x[:2]; v = x[2:]
    M = M_of_q(q, prm)
    return 0.5*v.T @ (M @ v) + 0.5*prm.k_psi*np.linalg.norm(q - prm.qg)**2

def simulate(prm: Params, law: str, q0, v0, h=0.001, tmax=20.0, tol=1e-3):
    x = np.hstack([q0, v0])
    Xs=[x.copy()]; Ts=[0.0]; Es=[energy(x, prm)]
    Eprev = energy(x, prm)
    for k in range(int(tmax/h)):
        print(f"{k} / {int(tmax/h)}", end="\r")
        d,_,_ = dist_n_t(x[:2], prm.obs)
        if d < 0: break
        x = rk4(x, h, rhs_second_order, prm, law)
        E = energy(x, prm)
        if E > Eprev + 1e-8: # Check if energy decrease is violated
            pass
        Eprev = E
        Xs.append(x.copy()); Ts.append(Ts[-1]+h); Es.append(energy(x, prm))
        if np.linalg.norm(x[:2] - prm.qg) < tol: break
    return np.array(Ts), np.array(Xs), np.array(Es)


# Metrics for sweep
def trajectory_metrics(prm: Params, law: str, starts, h=1e-3, tmax=20.0, tol=1e-3):
    min_d_all=[]; converged=[]; t_goal=[]; over_idx=[]; curv95=[]
    for q0 in starts:
        print(f"q0 = {q0}")
        d, n, t = dist_n_t(q0, prm.obs); v0 = 0.35*t  # show underdamping
        _, Xs, _ = simulate(prm, law, q0, v0, h=h, tmax=tmax, tol=tol)
        Q = Xs[:,:2]; V = Xs[:,2:]; A = np.array([rhs_second_order(x,prm,law)[2:] for x in Xs])

        # min distance
        ds = np.array([max(dist_n_t(q,prm.obs)[0], -np.inf) for q in Q])
        min_d_all.append(np.min(ds))

        # convergence time
        hit = np.where(np.linalg.norm(Q - prm.qg, axis=1) < tol)[0]
        converged.append(len(hit)>0)
        t_goal.append( (hit[0]*h) if len(hit)>0 else np.inf )

        # underdamping index: overshoot ratio on radial distance to goal
        rg = np.linalg.norm(Q - prm.qg, axis=1)
        if len(hit)>0:
            rmax = np.max(rg[:hit[0]+1]); rinit = rg[0]
        else:
            rmax = np.max(rg); rinit = rg[0]
        over = (rmax - rinit)/max(rinit,1e-6)
        over_idx.append(over)

        # curvature |κ|
        speed = np.linalg.norm(V,axis=1); cross = V[:,0]*A[:,1]-V[:,1]*A[:,0]
        kap = np.zeros_like(speed); nz = speed>1e-6
        kap[nz] = np.abs(cross[nz]/(speed[nz]**3))
        curv95.append(np.nanpercentile(kap, 95))

    # aggregate
    return {
        'min_d': float(np.min(min_d_all)),
        'conv_rate': float(np.mean(converged)),
        't_goal_mean': float(np.mean([tg for tg in t_goal if np.isfinite(tg)]) if np.any(np.isfinite(t_goal)) else np.inf),
        'overshoot_mean': float(np.mean(over_idx)),
        'curv95_mean': float(np.mean(curv95)),
    }


# Boundary compliance at grazing (2nd-order A-figure metric) 
# # boundary test: at ring distance d, grazing (n·v=0), check n·a>=0 for several v_t magnitudes
def boundary_compliance(prm: Params, law: str, Ds=None, vt_list=None):
    Ds = np.geomspace(0.01, 0.35, 16) if Ds is None else Ds
    vt_list = [0.2,0.5,0.8] if vt_list is None else vt_list
    thetas = np.linspace(0, 2*np.pi, 180, endpoint=False)
    fracs=[]
    for i, d in enumerate(Ds):
        print(f"{i} / {len(Ds)}", end='\r')
        ok=0; tot=0
        for th in thetas:
            q = prm.obs.c + (prm.obs.r + d)*np.array([np.cos(th), np.sin(th)])
            dd, n, t = dist_n_t(q, prm.obs)
            for vt in vt_list:
                v = vt*t
                M = M_of_q(q, prm); Minv = np.linalg.inv(M)
                Cv = C_times_qdot(q, v, prm); N = N_of_q(q, prm, law)
                a = Minv @ (N@v - Cv - prm.c_damp*(M@v) - grad_psi(q,prm))
                ok += 1 if (n @ a >= -1e-9) else 0
                tot += 1
        fracs.append(ok/tot)
    return float(np.mean(fracs))

# Sweep
def generate_param_tuples(alphas, ps, epses, kBs, grid=True, random_n=40):
    if grid:
        for a,p,e,k in itertools.product(alphas, ps, epses, kBs):
            yield (a,p,e,k)
    else:
        # Latin-ish random sampling
        pool=[]
        for _ in range(random_n):
            pool.append((random.choice(alphas),
                         random.choice(ps),
                         random.choice(epses),
                         random.choice(kBs)))
        for t in pool: yield t


def rank_and_select(rows):
    # Hard constraints first: min_d >= 0 (no collision), conv_rate == 1.0
    feasible = [r for r in rows if r['min_d']>=0 and r['conv_rate']>=0.999]
    if not feasible:
        # relax slightly: min_d >= -1e-3
        feasible = [r for r in rows if r['min_d']>=-1e-3 and r['conv_rate']>=0.9]
    if not feasible:
        # worst-case fallback
        feasible = rows[:]
    # Lexicographic: max boundary compliance -> min t_goal_mean -> min curv95_mean -> min overshoot
    feasible.sort(key=lambda r:(-r['bndry_comp'], r['t_goal_mean'], r['curv95_mean'], r['overshoot_mean']))
    return feasible[0]




# curvature series: |v x a|/|v|^3
def curvature_series(Xs, prm: Params, law: str):
    ks=[]; ds=[]
    for x in Xs:
        q = x[:2]; v = x[2:]
        a = rhs_second_order(x, prm, law)[2:]
        sp = np.linalg.norm(v)
        if sp < 1e-6: ks.append(0.0)
        else:
            cross = v[0]*a[1]-v[1]*a[0]
            ks.append(abs(cross)/(sp**3))
        d,_,_ = dist_n_t(q, prm.obs)
        ds.append(max(d,0.0))
    return np.array(ds), np.array(ks)


# Acceleration stream background (2nd-order visualization)
def acc_field_on_grid(prm: Params, law: str, vt_stream=0.6, Nx=60, Ny=60):
    xs = np.linspace(XMIN, XMAX, Nx); ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    U = np.zeros_like(XX); V = np.zeros_like(YY)
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j,i], YY[j,i]])
            d, n, t = dist_n_t(q, prm.obs)
            if d <= 0: continue
            v_graze = vt_stream * t  # grazing speed to expose B-effects
            a = rhs_second_order(np.hstack([q, v_graze]), prm, law)[2:]
            U[j,i], V[j,i] = a
    return XX, YY, U, V


# -----------------------------
# FIG A: Invariance vs d (second-order grazing test) + worst-case n·a
# -----------------------------
def figA_invariance(prm: Params, save_as):
    print("Figure A")
    Ds = np.geomspace(1e-3, 0.35, 32)
    thetas = np.linspace(0, 2*np.pi, 240, endpoint=False)
    vt_list = [0.2, 0.5, 0.8]   # tangential speeds to test grazing
    modes = {
        'no metric & no mag':    {'alpha':0.0, 'law':'none'},
        'metric only':           {'alpha':prm.alpha, 'law':'none'},
        'metric+mag const':      {'alpha':prm.alpha, 'law':'const'},
        'metric+mag d^{p-1}':    {'alpha':prm.alpha, 'law':'dpminus1'},
        'metric+mag d^{p}':      {'alpha':prm.alpha, 'law':'dp'},
        'metric+mag TAN d^{p}':  {'alpha':prm.alpha, 'law':'tan'},
    }
    frac = {k: [] for k in modes}
    min_na = {k: [] for k in modes}

    for i, d in enumerate(Ds):
        print(f"{i} / {len(Ds)}", end='\r')
        for name, cfg in modes.items():
            prm_loc = Params(**{**prm.__dict__, 'alpha':cfg['alpha']})
            ok=0; tot=0; mins=[]
            for th in thetas:
                q = prm.obs.c + (prm.obs.r + d)*np.array([np.cos(th), np.sin(th)])
                dd, n, t = dist_n_t(q, prm.obs)
                for vt in vt_list:
                    v = vt*t  # grazing
                    M = M_of_q(q, prm_loc); Minv=np.linalg.inv(M)
                    Cv = C_times_qdot(q, v, prm_loc); N=N_of_q(q, prm_loc, cfg['law'])
                    a = Minv @ (N@v - Cv - prm_loc.c_damp*(M@v) - grad_psi(q,prm_loc))
                    na = float(n @ a)
                    mins.append(na)
                    ok += 1 if (na >= -1e-9) else 0
                    tot += 1
            frac[name].append(ok/tot)
            min_na[name].append(np.min(mins))

    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for name in modes:
        axes[0].plot(Ds, frac[name], label=name)
    axes[0].set_xscale('log'); axes[0].set_ylim([0,1.05])
    axes[0].set_xlabel('distance d'); axes[0].set_ylabel('fraction with n·a ≥ 0 (grazing)')
    axes[0].set_title('Second-order boundary (Nagumo-like) — violations vs d')
    axes[0].grid(True, which='both', alpha=0.3); axes[0].legend(fontsize=8)

    for name in modes:
        axes[1].plot(Ds, min_na[name], label=name)
    axes[1].set_xscale('log'); axes[1].set_xlabel('distance d'); axes[1].set_ylabel('min n·a on ring (grazing)')
    axes[1].set_title('Worst-case boundary pointing (acceleration)')
    axes[1].grid(True, which='both', alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout(); 
    if SAVE:
        plt.savefig(save_as, dpi=180); plt.close(fig)
    else:
        plt.show()


# -----------------------------
# FIG B: Trajectories by mode + acceleration streamlines (grazing)
# -----------------------------
def figB_trajectories(prm: Params, save_as):
    print("Figure B")
    modes = [
        ('no metric & no mag',   {'alpha':0.0,        'law':'none'}),
        ('metric only',          {'alpha':prm.alpha,  'law':'none'}),
        ('metric+mag const',     {'alpha':prm.alpha,  'law':'const'}),
        ('metric+mag d^{p-1}',   {'alpha':prm.alpha,  'law':'dpminus1'}),
        ('metric+mag d^{p}',     {'alpha':prm.alpha,  'law':'dp'}),
        ('metric+mag TAN d^{p}', {'alpha':prm.alpha,  'law':'tan'}),
    ]
    starts = [np.array([-1.6,-1.4]), np.array([-1.4,0.6]),
              np.array([ 0.6,-1.5]), np.array([-1.0,1.6])]
    # fig, axes = plt.subplots(int(np.ceil(len(modes)/2)), 2, figsize=(4.6*len(modes), 4.2))
    fig, axes = plt.subplots(1, len(modes), figsize=(4.6*len(modes), 4.2))

    for ax, (title, cfg) in zip(axes, modes):
        print(f"law: {cfg['law']}")
        prm_loc = Params(**{**prm.__dict__, 'alpha':cfg['alpha']})
        # background: acceleration field at grazing speed (so B shows)
        XX,YY,U,V = acc_field_on_grid(prm_loc, cfg['law'], vt_stream=0.6, Nx=60, Ny=60)
        sp = np.sqrt(U**2+V**2)
        ax.streamplot(XX,YY,U,V, color=np.clip(sp,0,3), density=1.0, linewidth=0.6, cmap='viridis')

        # obstacle & goal
        ax.add_patch(Circle(prm.obs.c, prm.obs.r, color='k', alpha=0.15))
        ax.plot(prm.qg[0], prm.qg[1], 'r*', ms=12)

        # trajectories (second order; small tangential v0 to show underdamping)
        for q0 in starts:
            print(f"q0 = {q0}")
            d, n, t = dist_n_t(q0, prm.obs)
            v0 = 0.35*t
            _, Xs, _ = simulate(prm_loc, cfg['law'], q0, v0, h=0.05, tmax=20.0)
            Qs = Xs[:,:2]
            ax.plot(Qs[:,0], Qs[:,1], '-', lw=2)
            ax.plot(q0[0], q0[1], 'ko', ms=3)

        ax.set_title(title, fontsize=10)
        scaling = 2
        ax.set_xlim([XMIN*scaling,XMAX*scaling]); ax.set_ylim([YMIN*scaling,YMAX*scaling]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    plt.tight_layout(); 
    if SAVE:
        plt.savefig(save_as, dpi=180); plt.close(fig)
    else:
        plt.show()


# -----------------------------
# FIG C: Ring-averaged accelerative analogs: <max(0,n·a)> and <|t·a|>
# -----------------------------
def figC_ring_accels(prm: Params, save_as):
    print("Figure C")
    Ds = np.geomspace(1e-3, 0.35, 32)
    thetas = np.linspace(0, 2*np.pi, 240, endpoint=False)
    vt = 0.6  # canonical grazing tangential speed
    modes = {
        'no metric & no mag':    {'alpha':0.0, 'law':'none'},
        'metric only':           {'alpha':prm.alpha, 'law':'none'},
        'metric+mag const':      {'alpha':prm.alpha, 'law':'const'},
        'metric+mag d^{p-1}':    {'alpha':prm.alpha, 'law':'dpminus1'},
        'metric+mag d^{p}':      {'alpha':prm.alpha, 'law':'dp'},
        'metric+mag TAN d^{p}':  {'alpha':prm.alpha, 'law':'tan'},
    }
    avg_na = {k: [] for k in modes}
    avg_ta = {k: [] for k in modes}

    for d in Ds:
        for name, cfg in modes.items():
            prm_loc = Params(**{**prm.__dict__, 'alpha':cfg['alpha']})
            nas=[]; tas=[]
            for th in thetas:
                q = prm.obs.c + (prm.obs.r + d)*np.array([np.cos(th), np.sin(th)])
                dd, n, t = dist_n_t(q, prm.obs)
                v = vt*t
                M = M_of_q(q, prm_loc); Minv=np.linalg.inv(M)
                Cv = C_times_qdot(q, v, prm_loc); N=N_of_q(q, prm_loc, cfg['law'])
                a = Minv @ (N@v - Cv - prm_loc.c_damp*(M@v) - grad_psi(q,prm_loc))
                nas.append(max(0.0, float(n @ a)))
                tas.append(abs(float(t @ a)))
            avg_na[name].append(np.mean(nas))
            avg_ta[name].append(np.mean(tas))

    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for name in modes:
        axes[0].plot(Ds, avg_na[name], label=name)
    axes[0].set_xscale('log'); axes[0].set_title('⟨max(0, n·a)⟩ on rings (grazing)')
    axes[0].set_xlabel('distance d'); axes[0].set_ylabel('outward normal accel.')
    axes[0].grid(True, which='both', alpha=0.3); axes[0].legend(fontsize=8)

    for name in modes:
        axes[1].plot(Ds, avg_ta[name], label=name)
    axes[1].set_xscale('log'); axes[1].set_title('⟨|t·a|⟩ on rings (grazing)')
    axes[1].set_xlabel('distance d'); axes[1].set_ylabel('tangential accel.')
    axes[1].grid(True, which='both', alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout(); 
    if SAVE:
        plt.savefig(save_as, dpi=180); plt.close(fig)
    else:
        plt.show()


# -----------------------------
# FIG D: Curvature maps |κ| for full vs TAN, using grazing speed
# -----------------------------
def figD_curvature_maps(prm: Params, save_as):
    print("Figure D")
    def kappa_max_over_vt(q, prm_loc, law, vts=(0.3, 0.6, 0.9)):
        d, n, t = dist_n_t(q, prm_loc.obs)
        if d <= 0: return np.nan
        vals = []
        for vt in vts:
            v = vt*t
            a = rhs_second_order(np.hstack([q, v]), prm_loc, law)[2:]
            sp = np.linalg.norm(v)
            if sp < 1e-8: vals.append(0.0)
            else:
                cross = v[0]*a[1]-v[1]*a[0]
                vals.append(abs(cross)/(sp**3))
        return max(vals)

    Nx, Ny = 90, 90
    xs = np.linspace(XMIN, XMAX, Nx); ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    prm_full = Params(**prm.__dict__)
    prm_tan  = Params(**prm.__dict__)
    prm_const = Params(**prm.__dict__)

    K_full = np.full_like(XX, np.nan, dtype=float)
    K_tan  = np.full_like(XX, np.nan, dtype=float)
    K_const = np.full_like(XX, np.nan, dtype=float)

    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j,i], YY[j,i]])
            K_full[j,i] = kappa_max_over_vt(q, prm_full, 'dp')
            K_tan[j,i]  = kappa_max_over_vt(q, prm_tan,  'tan')
            K_const[j,i]  = kappa_max_over_vt(q, prm_const,  'const')

    
    vmax = np.nanpercentile(np.hstack([np.abs(K_full).ravel(), np.abs(K_tan).ravel()]), 99)
    vmax = max(vmax, 1e-3)

    fig, axes = plt.subplots(1,3, figsize=(17,5))
    im0 = axes[0].imshow(np.abs(K_full), origin='lower', extent=[XMIN,XMAX,YMIN,YMAX],
                         cmap='magma', vmin=0, vmax=vmax)
    axes[0].add_patch(Circle(prm.obs.c, prm.obs.r, color='w', alpha=0.6))
    axes[0].plot(prm.qg[0], prm.qg[1], 'c*', ms=12)
    axes[0].set_title('|κ|: metric + mag (b~d^p, full)'); axes[0].set_aspect('equal')

    im1 = axes[1].imshow(np.abs(K_tan), origin='lower', extent=[XMIN,XMAX,YMIN,YMAX],
                         cmap='magma', vmin=0, vmax=vmax)
    axes[1].add_patch(Circle(prm.obs.c, prm.obs.r, color='w', alpha=0.6))
    axes[1].plot(prm.qg[0], prm.qg[1], 'c*', ms=12)
    axes[1].set_title('|κ|: metric + mag TAN (b~d^p · |g_n^nat|)'); axes[1].set_aspect('equal')

    # Contrast vs const (full - const)
    diff = np.clip(np.abs(K_full) - np.abs(K_const), 
    a_min=-np.inf, a_max=np.inf)
    def d_of_q(q, obs): return np.linalg.norm(q - obs.c) - obs.r

    W = np.ones_like(K_full)
    for i in range(K_full.shape[1]):
        for j in range(K_full.shape[0]):
            q = np.array([XX[j,i], YY[j,i]])
            d = d_of_q(q, prm.obs)
            W[j,i] = np.exp(-(max(d,0.0)/0.35)**2)  # near-field emphasis

    diff_near = (np.abs(K_const) - np.abs(K_full)) * W  # positive=improvement near obstacle

    vmax2 = np.nanpercentile(np.abs(diff).ravel(), 99)
    # vmax2 = np.nanpercentile(np.abs(diff_near).ravel(), 99)
    vmax2 = max(vmax2, 1e-4)
    im2 = axes[2].imshow(diff, origin='lower', extent=[XMIN,XMAX,YMIN,YMAX],
                         cmap='coolwarm', vmin=-vmax2, vmax=vmax2)
    axes[2].add_patch(Circle(prm.obs.c, prm.obs.r, color='w', alpha=0.6))
    axes[2].plot(prm.qg[0], prm.qg[1], 'k*', ms=12)
    axes[2].set_title('Contrast: |κ|(d^p full) - |κ|(const)')
    axes[2].set_aspect('equal')

    plt.tight_layout(); 
    if SAVE:
        plt.savefig(save_as, dpi=180); plt.close(fig)
    else:
        plt.show()


# -----------------------------
# FIG F: Conservative safe k for b(d)=k d^p (second-order design)
# (same ring-based diagnostic but with G^{-1} from M(q))
# -----------------------------
def ksafe_from_metric(prm, d_min=0.0, d_max=1.0):
    Ds = np.geomspace(d_min, d_max, 24)
    thetas = np.linspace(0, 2*np.pi, 360, endpoint=False)
    kmax_d = []
    for d in Ds:
        ks = []
        for th in thetas:
            q = prm.obs.c + (prm.obs.r + d) * np.array([np.cos(th), np.sin(th)])
            Ginv, n, t, dd = Ginv_of_q(q, prm)
            gnat = Ginv @ (q - prm.qg)
            g_n = float(n @ gnat); g_t = abs(float(t @ gnat))
            if g_t>1e-12: ks.append(g_n / ((d**prm.p)*g_t))
        kmax_d.append(np.nan if len(ks)==0 else np.max([k for k in ks if k>0]))
    kmax_d = np.array(kmax_d)
    k_safe = 0.5*np.nanmin(kmax_d)
    return Ds, kmax_d, k_safe


def figF_kmax_safe(prm: Params, save_as, d_min=0.01, d_max=0.35):
    print('Figure F')
    Ds, kmax_d, k_safe = ksafe_from_metric(prm, d_min=d_min, d_max=d_max)

    fig, ax = plt.subplots(1,1, figsize=(6.4,4.2))
    ax.plot(Ds, kmax_d, 'o-', label='ring-wise k_max(d)')
    ax.axhline(k_safe, color='r', ls='--', label=f'suggested k_safe≈{k_safe:.3f}')
    ax.set_xscale('log'); ax.set_xlabel('distance d'); ax.set_ylabel('k_max(d)')
    ax.set_title('Conservative safe k for b(d)=k d^p (second order)')
    ax.grid(True, which='both', alpha=0.3); ax.legend()
    plt.tight_layout(); 
    if SAVE:
        plt.savefig(save_as, dpi=180); plt.close(fig)
    else:
        plt.show()



# -----------------------------
# MAIN
# -----------------------------
def main(optimise=True, optimise2=True, filename=None):
    obs = Obstacle(c=np.array([0.0, 0.0]), r=0.5)
    base = Params(qg=np.array([1.2, 1.0]), obs=obs, m0=1.0, alpha=1.2, eps=0.05, p=2.0, c_damp=1.2, kB=1.0, k_psi=1.0)

    if optimise:
        alpha = [0.8,1.2,1.6]
        p = [1.5,2.0,2.5]
        eps = [0.03, 0.05, 0.08]
        kB = [0.6, 1.0, 1.4]
        grid = True
        rdm = 0
        tmax = 20.0 #s
        h = 0.05 # step size simulate
    
    
        starts = [np.array([-1.6,-1.4]), np.array([-1.4,0.6]),
                np.array([ 0.6,-1.5]), np.array([-1.0,1.6])]
        
        tuples = list(generate_param_tuples(alpha, p, eps, kB,
                                            grid=grid, random_n=rdm))
    
        rows = []
        for i, (a, p, e, k) in enumerate(tuples):
            print(f"{i}/{len(tuples)} a: {a} | p: {p} | e: {e} | k: {k}")
            prm = Params(**{**base.__dict__, 'alpha':a, 'p':p, 'eps':e, 'kB':k})
            # Evaluate two modes for metrics: metric-only vs metric+mag d^p (final)
            met  = trajectory_metrics(prm, 'none', starts, h=h, tmax=tmax)
            full = trajectory_metrics(prm, 'dp',   starts, h=h, tmax=tmax)
            bcomp= boundary_compliance(prm, 'dp')

            rows.append({
                'alpha':a,'p':p,'eps':e,'kB':k,'c_damp':base.c_damp,
                'min_d': min(met['min_d'], full['min_d']),
                'conv_rate': min(met['conv_rate'], full['conv_rate']),
                't_goal_mean': full['t_goal_mean'],       # optimize with mag
                'overshoot_mean': full['overshoot_mean'], # show some underdamping
                'curv95_mean': full['curv95_mean'],
                'bndry_comp': bcomp
            })

        # Save sweep CSV
        with open('sweep_results_SO.csv','w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        # Pick best
        best = rank_and_select(rows)
        with open('best_config.json','w') as f: json.dump(best,f,indent=2)
        print('Best config:', best)

    elif optimise2:
        d_star = 0.08        # design distance (m) where shield is specified
        d_min  = 0.01        # smallest ring for k_safe
        m0     = 1.0
        eps    = 0.10        # keep eps fixed or add it to the sweep if desired
        c_damp = 0.8         # moderate damping to show underdamping but ensure convergence

    
        # Sweep sets (playbook knobs)
        LAMBDA = [4, 6, 8, 10, 12]  # shield ratios at d_star
        P_LIST = [1.5, 2.0, 2.5]    # decay exponents for b(d)=k d^p (or TAN)
        ETA    = [0.02, 0.05, 0.10] # fraction of conservative safe k

        starts = [
            np.array([-1.6, -1.4]), np.array([-1.4, 0.6]),
            np.array([ 0.6, -1.5]), np.array([-1.0, 1.6])
        ]

        rows = []
        for i, (L, p, eta) in enumerate(itertools.product(LAMBDA, P_LIST, ETA)):
            print(f"{i}/{len(itertools.product(LAMBDA, P_LIST, ETA))} lambda: {L} | p: {p} | eta: {eta}")
            # Map (L, eps, d_star) to alpha
            alpha = (L - 1.0) * m0 * (d_star**2 + eps**2)

            # Build params (kB will be filled after we compute safe k)
            prm_tmp = Params(qg=np.array([1.2, 1.0]),
                            obs=Obstacle(c=np.array([0.0,0.0]), r=0.5),
                            m0=m0, alpha=alpha, eps=eps, p=p, c_damp=c_damp, kB=0.0)

            # Conservative safe k from Fig-F routine restricted to [d_min, d_star]
            _, _, k_safe = ksafe_from_metric(prm_tmp, d_min=d_min, d_max=d_star)

            # Set magnetic gain from fraction eta of conservative bound
            prm_tmp.kB = eta * k_safe

            # Evaluate metrics with and without magnet
            met  = trajectory_metrics(prm_tmp, 'none', starts, h=0.05, tmax=20.0)
            full = trajectory_metrics(prm_tmp, 'dp',   starts, h=0.05, tmax=20.0)
            bcomp= boundary_compliance(prm_tmp, 'dp')

            rows.append({
                'Lambda':L, 'p':p, 'eta':eta, 'alpha':alpha,
                'eps':eps, 'kB':prm_tmp.kB, 'c_damp':c_damp,
                'min_d': min(met['min_d'], full['min_d']),
                'conv_rate': min(met['conv_rate'], full['conv_rate']),
                't_goal_mean': full['t_goal_mean'],
                'overshoot_mean': full['overshoot_mean'],
                'curv95_mean': full['curv95_mean'],
                'bndry_comp': bcomp
            })

        
        # Save & pick best (same selection as before)
        with open('sweep_results_Lambda.csv','w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        best = rank_and_select(rows)  # your lexicographic filter
        with open('best_config_Lambda.json','w') as f: json.dump(best,f,indent=2)
        print('Best config:', best)

    elif filename is not None: 
        with open(filename,'r') as f: 
            best = json.load(f)

        
    if optimise or filename is not None:   
        # Re-generate A–F with best config
        prm_best = Params(**{**base.__dict__,
                            'alpha':best['alpha'], 'p':best['p'],
                            'eps':best['eps'], 'kB':best['kB']})
    else:
        prm_best = base

    figA_invariance(prm_best, 'figs/figA_invariance_rings_SO.png')
    figB_trajectories(prm_best,'figs/figB_trajectories_modes_SO.png')
    figC_ring_accels(prm_best,'figs/figC_ring_accels_SO.png')
    figD_curvature_maps(prm_best,'figs/figD_curvature_dp_SO.png')
    figF_kmax_safe(prm_best,'figs/figF_kmax_safe_SO.png')
    print('Generated second-order figures: A, B, C, D, F')

if __name__ == "__main__":
    # main(optimise=False, optimise2=False, filename="best_config.json")
    main(optimise=False, optimise2=False, filename="best_config_Lambda.json")
