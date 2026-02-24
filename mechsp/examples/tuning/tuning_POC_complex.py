import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass

# -----------------------------
# Scenario setup
# -----------------------------
SAVE = False # global value; save figures or show them

np.random.seed(0)

@dataclass
class Obstacle:
    c: np.ndarray  # center
    r: float       # radius

@dataclass
class Scenario:
    qg: np.ndarray
    obs: Obstacle
    mu: float = 0.1 #0.6        # weight of log barrier in potential
    sigma: float = 0.2     # scale for metric anisotropy
    p: float = 2.0         # exponent for metric normal amplification (>1)
    beta0: float = 1.0     # baseline magnetic gain (for constant-beta case)
    beta_mode: str = 'const'  # 'const' or 'prop_d' (beta ~ d)
    beta_prop_k: float = 2.0  # if prop_d, beta = k * d (near obstacle)

sc = Scenario(qg=np.array([1.2, 1.0]), obs=Obstacle(c=np.array([0.0, 0.0]), r=0.5))

# Workspace box for plotting
XMIN, XMAX, YMIN, YMAX = -2.0, 2.0, -2.0, 2.0

# -----------------------------
# Geometry helpers
# -----------------------------

J = np.array([[0.0, -1.0], [1.0, 0.0]])  # +90° rotation

def dist_to_obstacle(q, obs: Obstacle):
    # signed distance (outside positive)
    return np.linalg.norm(q - obs.c) - obs.r

def unit_normal_to_obstacle(q, obs: Obstacle):
    v = q - obs.c
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return np.array([1.0, 0.0])
    return v / nrm

# -----------------------------
# Potential and metric shaping
# -----------------------------

def potential_grad(q, sc: Scenario):
    """∇ψ(q) = (q - qg) - mu * ∇log d, with d clipped away from 0."""
    d = max(dist_to_obstacle(q, sc.obs), 1e-4)
    n = unit_normal_to_obstacle(q, sc.obs)
    grad_goal = (q - sc.qg)
    grad_barrier = -(sc.mu / d) * n  # derivative of -mu * log d
    return grad_goal + grad_barrier


def metric_matrix(q, sc: Scenario):
    """Riemannian metric G(q) that amplifies cost along obstacle normal.
    G = R^T diag(λ_n, λ_t) R, where R aligns with [t, n].
    We return G and its inverse.
    """
    n = unit_normal_to_obstacle(q, sc.obs)
    t = J @ n  # tangent (90 deg)
    d = max(dist_to_obstacle(q, sc.obs), 1e-4)

    lam_n = 1.0 + (sc.sigma / d) ** sc.p  # amplify normal direction as d→0
    lam_t = 1.0

    # Orthonormal basis [t, n]
    R = np.column_stack([t, n])
    D = np.diag([lam_t, lam_n])
    G = R @ D @ R.T
    Ginv = R @ np.diag([1.0/lam_t, 1.0/lam_n]) @ R.T
    return G, Ginv, lam_n, lam_t


def magnetic_gain(q, sc: Scenario):
    d = max(dist_to_obstacle(q, sc.obs), 1e-4)
    if sc.beta_mode == 'const':
        return sc.beta0
    elif sc.beta_mode == 'prop_d':
        # beta ~ k * d (keeps curvature bounded near obstacle)
        return sc.beta_prop_k * d
    else:
        return sc.beta0

# -----------------------------
# Vector fields for 4 cases
# -----------------------------

def v_baseline(q, sc: Scenario):
    g = potential_grad(q, sc)
    return -g


def v_metric(q, sc: Scenario):
    g = potential_grad(q, sc)
    G, Ginv, _, _ = metric_matrix(q, sc)
    return -(Ginv @ g)


def v_magnetic_only(q, sc: Scenario):
    g = potential_grad(q, sc)
    beta = magnetic_gain(q, sc)
    return -(g - beta * (J @ g))  # -g + beta J g


def v_metric_magnetic(q, sc: Scenario):
    g = potential_grad(q, sc)
    G, Ginv, _, _ = metric_matrix(q, sc)
    gnat = Ginv @ g
    beta = magnetic_gain(q, sc)
    return -(gnat - beta * (J @ gnat))

# Map names to functions
VF = {
    'Euclidean (baseline)': v_baseline,
    'Metric only': v_metric,
    'Magnetic only': v_magnetic_only,
    'Metric + Magnetic': v_metric_magnetic,
}

# -----------------------------
# Integrator and diagnostics
# -----------------------------

def rk4_step(q, f, h):
    k1 = f(q)
    k2 = f(q + 0.5*h*k1)
    k3 = f(q + 0.5*h*k2)
    k4 = f(q + h*k3)
    return q + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def simulate_trajectory(q0, vf_fun, sc: Scenario, h=0.01, tmax=25.0, goal_tol=1e-2):
    qs = [q0.copy()] ; ts = [0.0]
    ds = []        # distance to obstacle
    vns = []       # normal component of velocity
    ks = []        # curvature along trajectory
    speeds = []

    def f(q):
        return vf_fun(q, sc)

    t = 0.0
    max_steps = int(tmax / h)
    for k in range(max_steps):
        q = qs[-1]
        v = f(q)
        speed = np.linalg.norm(v)
        if speed < 1e-8:
            break
        # curvature: a = J_v(q) v, approximate J_v by finite differences
        eps = 1e-4
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        dv_dx = (f(q + eps*e1) - f(q - eps*e1)) / (2*eps)
        dv_dy = (f(q + eps*e2) - f(q - eps*e2)) / (2*eps)
        Jv = np.column_stack([dv_dx, dv_dy])
        a = Jv @ v
        # signed curvature in 2D
        cross = v[0]*a[1] - v[1]*a[0]
        kappa = cross / (speed**3)

        d = dist_to_obstacle(q, sc.obs)
        n = unit_normal_to_obstacle(q, sc.obs)
        vn = np.dot(v, n)  # normal comp (positive outward)

        qs.append(rk4_step(q, f, h))
        ts.append(t + h)
        ds.append(max(d, 1e-6))
        vns.append(abs(vn))
        ks.append(kappa)
        speeds.append(speed)

        t += h
        if np.linalg.norm(qs[-1] - sc.qg) < goal_tol:
            break
        # safety: stop if gets too close to obstacle
        if d < 1e-3:
            break

    return np.array(ts), np.array(qs), np.array(ds), np.array(vns), np.array(ks), np.array(speeds)


# Multiple initial conditions
inits = [np.array([-1.6, -1.4]),
         np.array([-1.5,  0.0]),
         np.array([-1.0,  1.5]),
         np.array([ 0.5, -1.5])]

# -----------------------------
# Run simulations for 4 cases
# -----------------------------
results = {}
for name, vf in VF.items():
    res_case = []
    for q0 in inits:
        ts, qs, ds, vns, ks, speeds = simulate_trajectory(q0, vf, sc, h=0.01, tmax=30.0)
        res_case.append({
            'q0': q0,
            'ts': ts, 'qs': qs, 'ds': ds, 'vns': vns, 'ks': ks, 'speeds': speeds
        })
    results[name] = res_case

# -----------------------------
# Figure 1: Trajectories for all 4 cases
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.ravel()

case_names = list(VF.keys())
for ax, name in zip(axes, case_names):
    # streamplot background
    Nx, Ny = 40, 40
    xs = np.linspace(XMIN, XMAX, Nx)
    ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    U = np.zeros_like(XX)
    V = np.zeros_like(YY)
    vf = VF[name]
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j, i], YY[j, i]])
            d = dist_to_obstacle(q, sc.obs)
            if d <= 0:  # inside obstacle, set to 0 to avoid drawing
                U[j, i] = 0
                V[j, i] = 0
            else:
                vv = vf(q, sc)
                U[j, i], V[j, i] = vv
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(XX, YY, U, V, color=np.clip(speed, 0, 3), density=1.2, linewidth=0.7, cmap='viridis')

    # obstacle and goal
    circ = Circle(sc.obs.c, sc.obs.r, color='k', alpha=0.15)
    ax.add_patch(circ)
    ax.plot(sc.qg[0], sc.qg[1], 'r*', markersize=14, label='goal')

    # trajectories
    for res in results[name]:
        qpath = res['qs']
        ax.plot(qpath[:,0], qpath[:,1], '-', lw=2)
        ax.plot(qpath[0,0], qpath[0,1], 'ko', ms=4)

    ax.set_title(name)
    ax.set_xlim([XMIN, XMAX])
    ax.set_ylim([YMIN, YMAX])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

fig.suptitle('Trajectories under 4 cases: baseline, metric, magnetic, metric+magnetic', fontsize=14)
plt.tight_layout(rect=[0,0,1,0.97])
if SAVE:
    fig.savefig('fig1_trajectories_4cases.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# -----------------------------
# Figure 2: Near-obstacle asymptotics (log-log |v_n| vs d)
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.ravel()

for ax, name in zip(axes, case_names):
    all_d = []
    all_vn = []
    for res in results[name]:
        d = res['ds']
        vn = res['vns']
        # focus near obstacle but positive distances
        mask = (d > 1e-3) & (d < 0.6)
        all_d.append(d[mask])
        all_vn.append(vn[mask])
    if len(all_d) == 0:
        continue
    dcat = np.concatenate(all_d)
    vncat = np.concatenate(all_vn)
    if len(dcat) > 10:
        # log-log fit
        x = np.log(dcat)
        y = np.log(np.clip(vncat, 1e-12, None))
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]  # vn ~ exp(b) * d^a
        ax.scatter(dcat, vncat, s=6, alpha=0.4)
        # fitted line
        d_line = np.linspace(dcat.min(), dcat.max(), 100)
        vn_fit = np.exp(b) * d_line**a
        ax.plot(d_line, vn_fit, 'r--', lw=2, label=f'fit slope ~ {a:.2f}')
        # theoretical slopes
        if name == 'Euclidean (baseline)' or name == 'Magnetic only':
            # expect ~ 1/d => slope -1
            ax.plot(d_line, (vn_fit[-1]/(d_line[-1]**-1)) * d_line**(-1), 'k:', lw=2, label='ref slope -1')
        if name == 'Metric only' or name == 'Metric + Magnetic':
            # expect ~ d^{p-1}
            p1 = sc.p - 1.0
            # scale to end point
            ax.plot(d_line, (vn_fit[-1]/(d_line[-1]**p1)) * d_line**(p1), 'k:', lw=2, label=f'ref slope p-1 = {p1:.1f}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('distance to obstacle d')
        ax.set_ylabel('|v_n| (normal speed)')
        ax.set_title(f'Near-obstacle asymptotics: {name}')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

plt.tight_layout()
if SAVE:
    fig.savefig('fig2_near_obstacle_asymptotics.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# -----------------------------
# Figure 3: Curvature maps (field curvature) + trajectories
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.ravel()

for ax, name in zip(axes, case_names):
    vf = VF[name]
    Nx, Ny = 80, 80
    xs = np.linspace(XMIN, XMAX, Nx)
    ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    Kmap = np.zeros_like(XX)

    def f(q):
        return vf(q, sc)

    # finite-diff Jacobians to compute curvature of streamlines at grid
    eps = 1e-4
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j, i], YY[j, i]])
            if dist_to_obstacle(q, sc.obs) <= 0:
                Kmap[j, i] = np.nan
                continue
            v = f(q)
            sp = np.linalg.norm(v)
            if sp < 1e-10:
                Kmap[j, i] = 0.0
                continue
            e1 = np.array([1.0, 0.0])
            e2 = np.array([0.0, 1.0])
            dv_dx = (f(q + eps*e1) - f(q - eps*e1)) / (2*eps)
            dv_dy = (f(q + eps*e2) - f(q - eps*e2)) / (2*eps)
            Jv = np.column_stack([dv_dx, dv_dy])
            a = Jv @ v
            cross = v[0]*a[1] - v[1]*a[0]
            kappa = cross / (sp**3)
            Kmap[j, i] = kappa

    # show absolute curvature magnitude
    im = ax.imshow(np.abs(Kmap), origin='lower', extent=[XMIN, XMAX, YMIN, YMAX], cmap='magma', vmin=0, vmax=np.nanpercentile(np.abs(Kmap), 99))
    circ = Circle(sc.obs.c, sc.obs.r, color='w', alpha=0.6)
    ax.add_patch(circ)
    ax.plot(sc.qg[0], sc.qg[1], 'c*', ms=12)

    for res in results[name]:
        qpath = res['qs']
        ax.plot(qpath[:,0], qpath[:,1], 'w-', lw=1.8)

    ax.set_title(f'|Curvature| map: {name}')
    ax.set_xlim([XMIN, XMAX]); ax.set_ylim([YMIN, YMAX]); ax.set_aspect('equal')

fig.colorbar(im, ax=axes, shrink=0.8, label='|κ| (streamline curvature)')
plt.tight_layout()
if SAVE:
    fig.savefig('fig3_curvature_maps.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# -----------------------------
# Figure 4: Convergence metrics & tuning heuristic (beta const vs beta ~ d)
# -----------------------------
def evaluate_case(sc_local: Scenario):
    # return total path length and time-to-goal for a representative init
    q0 = np.array([-1.6, -1.4])
    out = {}
    for name, vf in VF.items():
        ts, qs, ds, vns, ks, speeds = simulate_trajectory(q0, vf, sc_local, h=0.01, tmax=30.0)
        # path length
        diffs = np.diff(qs, axis=0)
        L = np.sum(np.linalg.norm(diffs, axis=1))
        T = ts[-1] if len(ts)>0 else np.nan
        out[name] = (L, T)
    return out

# Baseline sc (const beta)
out_const = evaluate_case(sc)
# Prop-d beta
sc_prop = Scenario(qg=sc.qg.copy(), obs=sc.obs, mu=sc.mu, sigma=sc.sigma, p=sc.p,
                   beta0=sc.beta0, beta_mode='prop_d', beta_prop_k=2.0)
out_prop = evaluate_case(sc_prop)

# Plot bar charts
labels = list(VF.keys())
L_const = [out_const[k][0] for k in labels]
T_const = [out_const[k][1] for k in labels]
L_prop = [out_prop[k][0] for k in labels]
T_prop = [out_prop[k][1] for k in labels]

x = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Path length
axes[0].bar(x - width/2, L_const, width, label='β = const')
axes[0].bar(x + width/2, L_prop, width, label='β ∝ d')
axes[0].set_title('Path length (single init)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=15)
axes[0].legend()
axes[0].grid(True, axis='y', alpha=0.3)
# Time-to-goal
axes[1].bar(x - width/2, T_const, width, label='β = const')
axes[1].bar(x + width/2, T_prop, width, label='β ∝ d')
axes[1].set_title('Time to reach goal (s)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=15)
axes[1].legend()
axes[1].grid(True, axis='y', alpha=0.3)
plt.tight_layout()
if SAVE:
    fig.savefig('fig4_tuning_const_vs_prop.png', dpi=180)
    plt.close(fig)
else:
    plt.show()

# -----------------------------
# Compute and print fitted slopes (for the main case Metric+Magnetic)
# -----------------------------
summary = {}
for name in case_names:
    all_d = []
    all_vn = []
    for res in results[name]:
        d = res['ds']; vn = res['vns']
        mask = (d > 1e-3) & (d < 0.6)
        all_d.append(d[mask])
        all_vn.append(vn[mask])
    if len(all_d) and len(np.concatenate(all_d))>5:
        dcat = np.concatenate(all_d)
        vncat = np.concatenate(all_vn)
        x = np.log(dcat)
        y = np.log(np.clip(vncat, 1e-12, None))
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        summary[name] = a

# Save a small text report
with open('report_asymptotics.txt', 'w') as f:
    f.write('Fitted log-log slopes for |v_n| vs d (near obstacle)\n')
    for k, a in summary.items():
        f.write(f'{k}: slope = {a:.2f}\n')

'Generated: fig1_trajectories_4cases.png, fig2_near_obstacle_asymptotics.png, fig3_curvature_maps.png, fig4_tuning_const_vs_prop.png, report_asymptotics.txt'