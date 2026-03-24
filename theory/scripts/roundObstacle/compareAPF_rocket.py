"""
Let's have some fun!

Side-by-side GIF:
  Left  : APF (attractive + repulsive) with U_rep background
  Right : 'Gyroscopic' (Magnetic) with signed magnetic curvature K_B background

Requirements:
  - navigate_roundObstacle.py  (latest ground-truth)
  - APF_roundObstacle.py       (with sign-correct, budgeted APF class)
  - rocket.png                 (transparent PNG).

Output:
  figs/avoidance_side_by_side.gif
"""


from __future__ import annotations
import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, transforms
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.image import imread

# import your toolbox
from navigate_roundObstacle import (
    Params, Obstacle, SecondOrderSystem,
    SignedPowerMagnetic, dist_n_t, _initial_positions_aligned_with_goal
)
from APF_roundObstacle import SecondOrderSystemAPF


# -------------------------------
# 1) Scenario / params
# -------------------------------
def make_params():
    obs = Obstacle(c=np.array([0.0, 0.0]), r=0.5)
    prm = Params(
        qg=np.array([1.20, 1.00]),
        obs=obs,
        m0=1.0,
        alpha=0.0, 
        eps=0.05,
        eps_b=0.10,
        p=2.0,
        c_damp=0.9,
        kB=700.0,
        k_psi=1.0,
        d_far=0.15,
        q_far=2.0,
    )
    return prm


# -------------------------------
# 2) Backgrounds
# -------------------------------
def grid_apf_Urep(apf_sys, xlim, ylim, Nx=160, Ny=160):
    xs = np.linspace(xlim[0], xlim[1], Nx)
    ys = np.linspace(ylim[0], ylim[1], Ny)
    XX, YY = np.meshgrid(xs, ys)
    U = np.full_like(XX, np.nan, dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j, i], YY[j, i]])
            d,_,_ = dist_n_t(q, apf_sys.prm.obs)
            if d <= 0: continue
            U[j, i] = apf_sys.U_rep(q)
    return XX, YY, U


def curvature_parts_at_q_Bonly(q, system, vt=0.6):
    """
    Compute signed curvature contribution from magnetic term only (B-panel logic):
      kappa_s,B = (v x a_B) / |v|^3,  a_B = M^{-1} N v
    Uses grazing direction v = vt * t(q).
    Returns np.nan inside obstacle.
    """
    d, n, t = dist_n_t(q, system.prm.obs)
    if d <= 0:
        return np.nan
    v = vt * t
    M = system.M_of_q(q); Minv = np.linalg.inv(M)
    N = system.N_of_q(q)
    aB = Minv @ (N @ v)
    sp = np.linalg.norm(v)
    if sp < 1e-10: return 0.0
    kB = (v[0]*aB[1] - v[1]*aB[0]) / (sp**3)
    return kB

def grid_KB(system, xlim, ylim, Nx=160, Ny=160, vt=0.6):
    xs = np.linspace(xlim[0], xlim[1], Nx)
    ys = np.linspace(ylim[0], ylim[1], Ny)
    XX, YY = np.meshgrid(xs, ys)
    KB = np.full_like(XX, np.nan, dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j, i], YY[j, i]])
            KB[j, i] = curvature_parts_at_q_Bonly(q, system, vt=vt)
    return XX, YY, KB


# -------------------------------
# 3) Simulate the two systems
# -------------------------------
def simulate_path(system: SecondOrderSystem, q0, v0, h=0.02, tmax=20.0):
    T, X, _ = system.simulate(q0, v0, h=h, tmax=tmax)
    X = np.asarray(X); T=np.asarray(T)
    Q = X[:,:2]; V = X[:,2:]
    return T, Q, V


def sample_to_grid(T, Q, V, t_grid, T_scale):
    """
    Sample trajectory onto common time grid (nearest index, then clamp).
    T_scale = T_max/max(t_grid)
    """
    idx = np.searchsorted(T/T_scale, t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(T)-1)
    return Q[idx], V[idx]



# -------------------------------
# 4) Helpers for rocket & trails
# -------------------------------
def load_rocket_png(path="rocket.png"):
    if os.path.exists(path):
        try:
            img = imread(path)
            return img
        except Exception:
            pass
    return None  # use fallback marker

class VortexRenderer:
    """
    Procedural swirling vortex confined inside a circle of radius r.
    One AxesImage is created and updated each frame; no Artists are added/removed.
    """
    def __init__(self, ax, center, r, *, res=320, n_arms=5, swirl=7.0,
                 col_dark=(0.10, 0.14, 0.22), col_bright=(0.60, 0.72, 1.00), alpha_scale=0.55, zorder=6):
        self.ax = ax
        self.cx, self.cy = center
        self.r = float(r)
        self.res = int(res)
        self.n_arms = float(n_arms)
        self.swirl = float(swirl)
        self.col_dark = np.array(col_dark, dtype=float)
        self.col_bright = np.array(col_bright, dtype=float)
        self.alpha_scale = float(alpha_scale)

        # Precompute a square grid centered at the hole, normalized to radius
        s = np.linspace(-self.r, self.r, self.res)
        self.X, self.Y = np.meshgrid(s, s)
        self.R = np.hypot(self.X, self.Y)
        self.Theta = np.arctan2(self.Y, self.X)
        self.mask_inside = (self.R <= self.r)

        # Prepare RGBA buffer and AxesImage once
        self._img = np.zeros((self.res, self.res, 4), dtype=float)
        self.artist = ax.imshow(self._img, origin='lower',
                                extent=[self.cx-self.r, self.cx+self.r, self.cy-self.r, self.cy+self.r],
                                interpolation='bilinear', zorder=zorder)
        # start fully transparent (so core circle from your obstacle remains visible)
        self._img[..., 3] = 0.0
        self.artist.set_data(self._img)
    
    
    def draw(self, phase):
        """
        Update the texture at given phase in [0, 2π).
        """
        rho = np.clip(self.R / self.r, 0.0, 1.0)            # 0 at center, 1 at boundary
        # Swirl pattern: sinusoidal spirals + radial swelling; phase animates rotation
        pat = 0.5 + 0.5 * np.sin(self.n_arms * self.Theta + self.swirl*(1.0 - rho) + phase)
        # emphasize mid radii, fade at center and at boundary
        # falloff = (1.0 - rho**2) * (0.65 + 0.35 * (1.0 - rho))
        falloff = 1.0
        brightness = pat * falloff

        # RGB blend between a dark base and a brighter blue; alpha scales with brightness
        rgb = (self.col_dark[None, None, :] * (0.6*brightness[..., None])
               + self.col_bright[None, None, :] * (0.4*brightness[..., None]))
        alpha = self.alpha_scale * brightness

        # Write into RGBA buffer (only inside the circle)
        img = self._img
        img[..., :3] = 0.0
        img[..., 3] = 0.0
        img[self.mask_inside, :3] = rgb[self.mask_inside]
        img[self.mask_inside,  3] = alpha[self.mask_inside]

        self.artist.set_data(img)


def make_trail_segments(Q):
    """
    Convert Q [N,2] into line segments [N-1, 2, 2]
    """
    if len(Q)<2: return np.zeros((0,2,2))
    P = Q.reshape(-1,1,2)
    return np.concatenate([P[:-1], P[1:]], axis=1)
    
def build_initial_conditions(
        prm, 
        n_aligned=5, 
        add_headon=True, 
        add_offsets=True):
    """
    Generate a small set of starts:
      - points on a segment 'behind' the obstacle (aligned with goal)
      - one head-on point on the goal axis (optional)
      - two lateral offsets (optional)
    """
    starts = _initial_positions_aligned_with_goal(prm, num=n_aligned).tolist()
    if add_headon:
        r2 = prm.obs.r + 0.85
        axis = (prm.qg - prm.obs.c)/np.linalg.norm(prm.qg - prm.obs.c)
        starts.append((prm.obs.c + r2*axis).tolist())
    if add_offsets:
        starts.append((np.array([-1.8, +0.2])).tolist())
        starts.append((np.array([-1.8, -0.6])).tolist())
    return np.array(starts)


# -------------------------------
# 5) Main animation
# -------------------------------
def main():
    os.makedirs("figs", exist_ok=True)
    prm = make_params()

    # --- Build systems ---
    apf = SecondOrderSystemAPF(prm,
                               k_att=prm.k_psi*0.5, 
                               d0=0.5, d_eps=0.03, U_wall=2.25, F_cap=30.0)  # tuned APF
    gyro = SecondOrderSystem(prm, SignedPowerMagnetic(prm))

    # --- Initial conditions
    starts = build_initial_conditions(prm, n_aligned=5, add_headon=False, add_offsets=False)
    vt_seed = 0.05

    # --- Common animation timing ---
    fps = 30
    dur_sec = 15.0
    h_int = 1.0/fps # integator step
    frames = int(fps*dur_sec)
    t_grid = np.linspace(0.0, dur_sec, frames)


    # --- Simulate ---
    Ts_apf, Qs_apf, Vs_apf, Ts_gy, Qs_gy, Vs_gy = [], [], [], [], [], []
    T_max = 0.0
    for i, q0 in enumerate(starts):
        print(f"{i + 1} / {len(starts)}: {q0}")
        d, n, t = dist_n_t(q0, prm.obs)
        v0 = 0.00*n + vt_seed*t
        T_a, Q_a, V_a = simulate_path(apf,  q0, v0, h=h_int, tmax=40.0)
        T_g, Q_g, V_g = simulate_path(gyro, q0, v0, h=h_int, tmax=40.0)
        Ts_apf.append(T_a); Ts_gy.append(T_g)
        Qs_apf.append(Q_a); Vs_apf.append(V_a)
        Qs_gy.append(Q_g);  Vs_gy.append(V_g)
        Tmax_temp = max(T_a[-1], T_g[-1])
        if Tmax_temp > T_max: T_max = Tmax_temp

    for i, (T, Q, V) in enumerate(zip(Ts_apf, Qs_apf, Vs_apf)):
        Qs_apf[i], Vs_apf[i] = sample_to_grid(T, Q, V, t_grid, T_max/dur_sec)
    for i, (T, Q, V) in enumerate(zip(Ts_gy, Qs_gy, Vs_gy)):
        Qs_gy[i], Vs_gy[i] = sample_to_grid(T, Q, V, t_grid, T_max/dur_sec)
    
    
    # If paths have different lengths, we map indices linearly to frames.
    idx_apf = np.linspace(0, len(t_grid)-1, frames).astype(int)
    idx_gy  = np.linspace(0, len(t_grid)-1,  frames).astype(int)

    # --- Backgrounds ---
    xlim=(-2.0, 2.0); ylim=(-2.0, 2.0)
    XX_L, YY_L, Urep = grid_apf_Urep(apf, xlim, ylim, Nx=200, Ny=200)
    XX_R, YY_R, KB   = grid_KB(gyro, xlim, ylim, Nx=200, Ny=200, vt=0.6)

    # Normalize backgrounds
    # U_rep is nonnegative; scale robustly
    Umax = np.nanmax(Urep)
    Unorm = Normalize(vmin=0.0, vmax=max(Umax, 1e-6))

    # Signed curvature: diverging around 0
    
    # Diverging map with black at zero:
    div_black = LinearSegmentedColormap.from_list(
        "div_black",
        [(0.00, "#1d4f8a"),   # deep blue at large negative curvature
        (0.50, "#000000"),   # **black** at zero curvature
        (1.00, "#ff6600")]   # warm red at large positive curvature
    )

    KB_max = max(np.nanmax(KB.ravel()), 1e-4)
    KB_min = min(np.nanmin(KB.ravel()), -1e-4)
    KBnorm = TwoSlopeNorm(vcenter=0.0, vmin=KB_min, vmax=KB_max)

    # Speed colormap for trails
    all_speeds = []
    for Vs in (Vs_apf, Vs_gy):
        for V in Vs:
            if len(V): all_speeds.append(np.linalg.norm(V, axis=1))

    smin = min(s.min() for s in all_speeds) if all_speeds else 0.0
    smax = max(s.max() for s in all_speeds) if all_speeds else 1.0
    speed_norm = Normalize(vmin=smin, vmax=smax)
    trail_cmap = plt.cm.plasma

    # Rocket sprite (optional)
    rocket_img = load_rocket_png("rocket.png")
    rocket_scale = 0.08  # image extent (meters) relative to plot

    # --- Figure and artists ---
    fig = plt.figure(figsize=(12.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.045, 1.0], wspace=0.06)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 1])

    vortexL = VortexRenderer(axL, center=prm.obs.c, r=prm.obs.r,
                         res=360, n_arms=5, swirl=7.0,
                         col_dark=(0.10,0.14,0.22), col_bright=(0.58,0.70,1.00), alpha_scale=0.55, zorder=6)
    vortexR = VortexRenderer(axR, center=prm.obs.c, r=prm.obs.r,
                         res=360, n_arms=5, swirl=7.0,
                         col_dark=(0.10,0.14,0.22), col_bright=(0.58,0.70,1.00), alpha_scale=0.55, zorder=6)
    
    
    # Keep (or re-add) the solid black core once; it stays under the vortex texture
    coreL = Circle(prm.obs.c, prm.obs.r, facecolor='k', edgecolor='k', zorder=5)
    coreR = Circle(prm.obs.c, prm.obs.r, facecolor='k', edgecolor='k', zorder=5)
    axL.add_patch(coreL); axR.add_patch(coreR)

    # Background images
    imL = axL.imshow(Urep, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                     cmap='magma', norm=Unorm, zorder=0)
    imR = axR.imshow(KB, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                     cmap=div_black, norm=KBnorm, zorder=0)
    
    def set_axes(ax, title):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect('equal'); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(prm.qg[0], prm.qg[1], marker='*', ms=14, color='#FFD000', zorder=20)
        ax.set_title(title, fontsize=12, pad=6)


    set_axes(axL, "APF")
    set_axes(axR, "Gyroscopic")



    # Trail line collections
    N = len(starts)
    # Rocket images
    rockets_L = [axL.imshow(np.zeros((2,2,4)), extent=[0,0,0,0], zorder=25) for _ in range(N)]
    rockets_R = [axR.imshow(np.zeros((2,2,4)), extent=[0,0,0,0], zorder=25) for _ in range(N)]

    
    segsL_all = [np.concatenate([Q[:-1,None,:], Q[1:,None,:]], axis=1) if len(Q)>1 else np.zeros((0,2,2))
                 for Q in Qs_apf]
    segsR_all = [np.concatenate([Q[:-1,None,:], Q[1:,None,:]], axis=1) if len(Q)>1 else np.zeros((0,2,2))
                 for Q in Qs_gy]
    speed_mid_L = [0.5*(np.linalg.norm(Vs_apf[i],axis=1)[:-1] + np.linalg.norm(Vs_apf[i],axis=1)[1:])
                   for i in range(N)]
    speed_mid_R = [0.5*(np.linalg.norm(Vs_gy[i], axis=1)[:-1] + np.linalg.norm(Vs_gy[i], axis=1)[1:])
                   for i in range(N)]

    lcs_L = [LineCollection([], cmap=trail_cmap, norm=speed_norm, linewidth=2.0, zorder=12) for _ in range(N)]
    lcs_R = [LineCollection([], cmap=trail_cmap, norm=speed_norm, linewidth=2.0, zorder=12) for _ in range(N)]
    for lc in lcs_L: axL.add_collection(lc); 
    for lc in lcs_R: axR.add_collection(lc)

    # Single shared colorbar for trail speed (place in the middle via inset axes)
    sm = plt.cm.ScalarMappable(cmap=trail_cmap, norm=speed_norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('speed |v| [m/s]')


    # ------------- animation update -------------
    def update(frame):
        # Progress counter
        pct = int(100*(frame+1)/frames)
        print(f"frame {frame+1}/{frames} ({pct:>3}%)", end='\r', flush=True)

        phase = 4*np.pi * (frame / frames)
        vortexL.draw(phase)
        vortexR.draw(phase + np.pi/4)

        def orient_and_place(ax, rocket, q, v, orientation_angle=0.0):
            if rocket_img is None:
                # fallback: small oriented triangle marker
                # draw as scatter; to keep it simple, use a simple dot if you prefer
                ax.plot(q[0], q[1], marker=(3, 0, math.degrees(math.atan2(v[1], v[0]))),
                        markersize=10, color='w', zorder=24)
                return
            # extent around (x,y)
            w = rocket_scale; h = rocket_scale
            rocket.set_data(rocket_img)
            rocket.set_extent([q[0]-w, q[0]+w, q[1]-h, q[1]+h])
            ang = math.atan2(v[1], v[0])  # radians; points along velocity
            tr = transforms.Affine2D().rotate_around(q[0], q[1], ang - orientation_angle) + ax.transData
            rocket.set_transform(tr)
            rocket.set_zorder(25)

        # Trails and rockets per IC
        for i in range(N):
            if frame >= 1:
                lcs_L[i].set_segments(segsL_all[i][:frame])
                lcs_R[i].set_segments(segsR_all[i][:frame])
                lcs_L[i].set_array(speed_mid_L[i][:frame])
                lcs_R[i].set_array(speed_mid_R[i][:frame])
                
            qL = Qs_apf[i][frame]; vL = Vs_apf[i][frame]
            qR = Qs_gy[i][frame];  vR = Vs_gy[i][frame]
            if np.linalg.norm(vL)<1e-6: vL = np.array([1e-6,0.0])
            if np.linalg.norm(vR)<1e-6: vR = np.array([1e-6,0.0])

            orient_and_place(axL, rockets_L[i], qL, vL, orientation_angle=45*np.pi/180)
            orient_and_place(axR, rockets_R[i], qR, vR, orientation_angle=45*np.pi/180)

        return rockets_L + rockets_R + lcs_L + lcs_R

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

    # Save GIF
    out_path = "figs/avoidance_side_by_side_multi_short.gif"
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
