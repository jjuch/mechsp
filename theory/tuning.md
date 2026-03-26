# Tuning for Safe and Agile Trajectory Shaping

> :warning: **This work is still under review and should not be used as a formal reference**
# Table of Contents
1. [Round Obstacles](#round-obstacle)
    1. [Euler–Lagrange with Levi–Civita and gyroscopic two‑form](#euler-lagrange)
    2. [Obstacle‑aware metric and two‑form](#obstacle-aware-metric)
    3. [How the Magnetic Field Shapes Curvature](#magnetic-field-curvature)
    4. [Magnetic Field Classes — Why So Many, and When to Use Which](#magnetic-classes)
        1. [`ConstMagnetic`  — simple far‑field limited constant](#cst-magn)
        2. [`PowerMagnetic` (`dp`) — near‑field emphasis](#power-magn)
        3. [`SignedPowerMagnetic` — head‑on analytic tool](#sgn-magn)
        4. [`SineMagnetic` — single‑lobe angular shaping with a near‑wall switch](#sine-magn)
        5. [`SineRPhaseMagnetic` — annular, radially phase‑swept sine](#swept-sine-magn)
        6. [`TiltedSineMagnetic` — biased annulus + sector window + optional knee‑cap](#tilt-magn)
    5. [APF Baseline and NF Context](#APF)
    6. [Diagnostics: What to Tune with What](#diagnostics)
    7. [Figure that can be generated](#figures)
    8. [References](#references)

# A. Round obstacles <a name="round-obstacle"></a>
We derive and tune a **second‑order** obstacle‑aware geometric controller **without** modifying the goal potential:

$$
\psi(q)=\tfrac12\|q-q_g\|^2,
$$

so all obstacle intelligence comes from:
- an **anisotropic metric** $M(q)$ that raises normal “inertia” near the boundary; and
- a **gyroscopic** two‑form $N(q)$ (skew) that **bends** trajectories but does **no work** on the energy.

This reference explains **how the magnetic (gyroscopic) field shapes curvature**, why we implement **multiple magnetic field classes**, and when that extra modeling **complexity** pays off. It also summarizes the **APF baseline**, standard **diagnostics**, and an explicit **figure checklist** for the _roundObstacle_ part of the repository.

### 1. Euler–Lagrange with Levi–Civita and gyroscopic two‑form <a name="euler-lagrange"></a>

We work on $\mathcal{Q}\subset\mathbb{R}^2$ with

$$
\begin{align}
&\mathcal{L}(q,\dot{q})=\tfrac12\,\dot q^\top M(q)\dot q + A(q)\cdot\dot q - \psi(q),\\ 
&B=dA,\\ 
&B(q)^\top=-B(q),
\end{align}
$$

and add Rayleigh damping $c_dM(q)\dot q$. The Euler–Lagrange equations are:

$$
\boxed{M(q)\ddot q + C(q,\dot q) \dot q + c_d M(q)\dot q + \nabla\psi(q) = N(q) \dot q,}
$$

with the **Levi–Civita** term

$$
\begin{align}
\big(C(q,\dot q)\dot q\big)^i &= \sum_{j,k}\Gamma^i_{jk}(q)\dot q^j\dot q^k,\\
\Gamma^i_{jk} &= \frac12\sum_\ell M^{i\ell}(\partial_j M_{\ell k}+\partial_k M_{\ell j}-\partial_\ell M_{jk}),
\end{align}
$$

and 

$$
N(q) = \begin{bmatrix} 0 & -B(q) \\ 
B(q) & 0\end{bmatrix}.
$$

Define energy $\mathcal{H}(q,\dot q)=\tfrac12\dot q^\top M(q)\dot q+\psi(q)$. Using the Levi–Civita identity and $B^\top=-B$, we get the **exact** balance

$$
\boxed{\dot{\mathcal{H}} = -c_d\dot q^\top M(q)\dot q \le 0,}
$$

i.e., geometry $C$ and gyroscopic $B$ **do not** change $\mathcal{H}$. Gyroscopic/skew terms do no work, they bend trajectories without altering energy. [[6]](https://link.springer.com/book/10.1007/978-1-4757-2063-1)

---

### 2. Obstacle‑aware metric and two‑form <a name="obstacle-aware-metric"></a>

For a disk obstacle with center $c$ and radius $r$, write

$$
\begin{cases}
d(q)=\|q-c\|-r,\\ 
n(q)=\frac{q-c}{\|q-c\|},\\ 
t(q)=J n(q),
\end{cases}
$$

with $J = \begin{bmatrix}0 & -1\\ 1 & 0\end{bmatrix}$ a 90° rotation.
We shape

$$
\boxed{M(q)=m_0 I_2+\alpha s(d(q)) n(q)n(q)^\top,\qquad s(d)=\frac{1}{d^2+\varepsilon^2}\;,}
$$

with $m_0>0$ the original mass of the agent, $\alpha>0$, $0<\varepsilon\ll r$. This increases the **normal** inertia $m_n=m_0+\alpha s(d)$ as $d\to 0^+$ while keeping the **tangential** inertia $m_t=m_0$ unchanged.

We implement the gyroscopic two‑form as

$$
\boxed{N(q)=B(q)J.}
$$

---
### 3. How the Magnetic Field Shapes Curvature<a name="magnetic-field-curvature"></a>

Let $\psi$ denote the **heading** of $v=\dot q$. Planar Frenet relations give (signed) curvature $\kappa_s$ via the heading rate: 

$$
\frac{d\psi}{dt} = \frac{v\times a}{\|v\|^2} = \kappa_s \|v\| \qquad \text{with }\kappa_s = \frac{d\psi}{ds}.
$$

These are standard and hold for any smooth planar trajectory.

#### 3.1 Magnetic contribution (Euclidean mass) <a name="magnetic-contrib"></a>
If $M(q)=m_0 I$, then the magnetic acceleration is

$$
 a_B = M^{-1} N v = \frac{b(q)}{m_0} J v.
$$

Therefore

$$
\begin{align}
 \kappa_{s,B} &= \frac{v\times a_B}{\|v\|^3}\\
 &= \frac{1}{\|v\|^3}\frac{b}{m_0} (v\times Jv)\\
 &= \frac{1}{\|v\|^3} \frac{b}{m_0} \|v\|^2\\
 &= \boxed{\;\frac{b(q)}{m_0 \|v\|}. }
\end{align}
$$

Equivalently, the **heading rate** contributed by the magnetic field obeys

$$
\boxed{\frac{d\psi_B}{dt} = \frac{b(q)}{m_0},}\qquad \text{so }\frac{d\psi_B}{ds} = \frac{b(q)}{m_0\|v\|}.
$$

**Implications** (visible in the curvature maps):
- **Scaling:** Curvature grows with $b(q)$ and decays with speed. At fixed grazing speed $v_t$, $\kappa_{s,B}=b/(m_0 v_t)$.
- **Sign:** $\text{sign}(\kappa_{s,B})=\text{sign}(b)$; laws that modulate the sign steer the bend direction.

When $\alpha>0$, $M(q)$ is no longer scalar. The code still builds curvature from **components** $a=a_B+a_{\rm geom}+a_{\rm goal}-c_{\rm damp}v$ via $(v\times a)/\|v\|^3$, rendering total, magnetic (B), geometric, and goal panels. See `plot_curvature_maps(...)`.

#### 3.2 Head‑on bending angle (exact and practical) <a name="headon-bending"></a>
For **head‑on** launch at radius $r_2$ with $\dot r(0)=-v_n$, and a law whose sign is consistent along the approach (e.g., `SignedPowerMagnetic`), combining $\dot r=\|v\|\cos\psi$ with the heading rate gives the exact separable ODE

$$
 \cos\psi d\psi = \frac{k_B}{m_0\|v\|} d^p \phi_{\rm far}(d) dd.
$$

Integrating from $\psi(0)=-\tfrac{\pi}{2}$ to **tangent** $\psi=0$ yields the exact threshold

$$
 \boxed{\frac{k_B}{m_0}\int_{0}^{T}\frac{d^p \phi_{\rm far}(d(t))}{\|v(t)\|} dt \ge 1. }
$$

With the constant‑speed approximation $\|v\|\approx v_n$, one gets $\tfrac{k_B}{m_0 v_n}I(d_2)\ge 1$, where $I(d_2)=\int_0^{d_2} d^p\phi_{\rm far}(d) dd$. The implementation improves this by measuring a **time‑dilation factor**

$$
 \eta = \Big\langle\frac{v_n}{\|v(d)\|}\Big\rangle_{f(d)} \ge 1, \qquad f(d)=d^p\phi_{\rm far}(d),
$$

using a **No‑Magnetic** head‑on run with identical damping/goal, and uses $\tfrac{k_B}{m_0 v_n}I(d_2)\ge \tfrac{1}{\eta}$ in design functions (`measure_eta`, `_I_of_d`, `_find_r2_for_kB`, `_kB_for_r2`).

> **Turning radius.** For Euclidean mass and speed $\|v\|$, the instantaneous magnetic turning radius is $\rho_B = 1/|\kappa_{s,B}| = (m_0 \|v\|)/|b|$ — a useful intuition for ring placement.

---

### 4. Magnetic Field Classes — Why So Many, and When to Use Which<a name="magnetic-classes"></a>

All classes implement $N(q)=b(q)J$ but differ in **where** and **how** they inject curvature. The hierarchy increases **spatial selectivity** (radial → angular → annular + sector + phase), which:
- concentrates curvature where it improves safety/economy,
- avoids over‑bending where it hurts path quality,
- closes well‑known failure corridors (e.g., **head‑on axes**), and
- reduces field “energy footprint” (smaller $\int b^2$ over the domain).

Below, “laws” refer to classes in `navigate_roundObstacle.py`. Symbols $\phi_{\rm far}(d)=\exp(-(d/d_{\rm on})^q)$, annular **Gaussian** $A(r)$, angular **window** $W(\theta)$, near‑wall **switch** $S(d)$, and radial **phase** $\phi(r)$ all appear in code.

#### 4.1 `ConstMagnetic`  — simple far‑field limited constant <a name="cst-magn"></a>
- **Law:** $b(d)=k_B \phi_{\rm far}(d)$.
- **When:** fast prototype; uniform curvature footprint, decays outside a chosen $d_{\rm on}$.
- **Pros:** few parameters; robust qualitative behavior.
- **Cons:** bends **everywhere** in the donut (no angular targeting) → unnecessary energy use.

#### 4.2 `PowerMagnetic` (`dp`) — near‑field emphasis<a name="power-magn"></a>
- **Law:** $b(d)=k_B (d+\varepsilon_b)^p \phi_{\rm far}(d)$, $p>1$.
- **When:** increase curvature near the wall and keep it modest outside.
- **Pros:** simple radial selectivity; good default.
- **Cons:** still no angular selectivity.

#### 4.3 `SignedPowerMagnetic` — head‑on analytic tool<a name="sgn-magn"></a>
- **Law:** $b(q)=k_B (d+\varepsilon_b)^p \phi_{\rm far}(d) \text{sign}(\sin\theta)$ with $\theta=\theta_{\rm rel\,goal}(q,q_g,c)$.
- **When:** design for **head‑on** worst case; sign stays consistent along the approach.
- **Pros:** enables **closed‑form** trade‑offs (`I(d_2)`) and **measured** $\eta$; perfect for minimal $r_2$ or minimal $k_B$ vs $v_n$.
- **Cons:** coarse angular control; good for certification, less for aesthetics.

#### 4.4 `SineMagnetic` — single‑lobe angular shaping with a near‑wall switch<a name="sine-magn"></a>
- **Law:** $b\propto d^p \phi_{\rm far}(d) \big[(1-S(d))+S(d)\sin\theta\big]$ with a smooth $S(d)$ that preserves outward sign right at the wall.
- **When:** suppress curvature on the **goal side** while protecting the wall.
- **Pros:** eliminates effort where it’s not needed; keeps near‑wall safeguarding.
- **Cons:** single lobe; no radius‑dependent phase.

#### 4.5 `SineRPhaseMagnetic` — annular, radially phase‑swept sine<a name="swept-sine-magn"></a>
- **Law:** 

$$
b(d,\theta,r)=k_B d^p w_{\rm ann}(r) \phi_{\rm far}(d) \sin\big(\theta+\phi(r)\big),
$$

with $\phi(r)$ sweeping from $-\phi_{\max}$ at $r_1$ to $+\phi_{\max}$ at $r_2$ and an angular window that **kills the donut on the goal side**.
- **When:** concentrate curvature in a **thin annulus** where it pays off; align bending direction across radii.
- **Pros:** very **clean B‑panel** on the hard semicircle; smaller energy footprint.
- **Cons:** more parameters (annulus, phase, window) but tunable via provided diagnostics.

#### 4.6 `TiltedSineMagnetic` — biased annulus + sector window + optional knee‑cap <a name="tilt-magn"></a>
- **Law:** $b_{\rm total}=b_{\rm tilted}+b_{\rm knee}$, with
  - $b_{\rm tilted}=k_B d^p \phi_{\rm far}(d) A(r) [ \varepsilon_0 + a_1(r)\sin(\theta+\phi(r)) ] W(\theta),$
  - a tiny **knee‑cap** bump of fixed sign just **outside** $r_2$ in a narrow wedge around the axis to close the **head‑on** corridor.
- **When:** you want **maximum control**: where (annulus), which sector (window), how much DC bias ($\varepsilon_0$), and a small **axis rescue**.
- **Pros:** simultaneously achieves: boundary **compliance** at grazing ($n\cdot a\ge0$ on the hard semicircle), strong **B‑panel** dominance in curvature, and robust **head‑on** deflection with modest $k_B$.
- **Cons:** highest parameter count; however each parameter has a clear **diagnostic** to tune it (below), turning complexity into **predictability** rather than trial‑and‑error.

> **Why complexity helps:** The curvature identity $\kappa_{s,B}=b/(m_0\|v\|)$ tells us exactly where to “spend” field strength. By adding radial (annulus), angular (window), phase (alignment), and axis patches (knee‑cap), we apply curvature **only** where it reduces collision risk or sharp corners — improving path quality, keeping speeds high, and lowering the overall actuation footprint. The provided predictors and maps quantify these effects directly.

---

### 5. APF Baseline and NF Context<a name="APF"></a>

**APF** (Artificial Potential Fields) guide motion by $-\nabla U=-(\nabla U_{\rm att}+\nabla U_{\rm rep})$. They are simple and reactive but may **slow down** (kinetic→potential→kinetic) and can form **local minima** near obstacles unless carefully shaped. The implementation here sets a **wall‑energy budget** $U_{\rm wall}\sim(1 \text{ – } 3)E_{\max}$ to derive $\eta_{\rm rep}$ and applies a smooth **force cap** to keep accelerations physical (no blow‑ups).

**NF** (Navigation Functions) are special artificial potentials with a **single** minimum at the goal (no spurious minima) on sphere‑worlds; following $-\nabla$ guarantees convergence and collision avoidance from almost all starts. They require more global structure than APF but avoid local traps by design.

---

### 6. Diagnostics: What to Tune with What<a name="diagnostics"></a>

- **Curvature maps** (`plot_curvature_maps`) — 4 panels (total, B, geometric, goal). Use them to ensure the **B‑panel** dominates in the hard semicircle and to see phase alignment across the annulus.
- **Grazing normal maps** (`plot_grazing_normal_maps`) — check $\langle n\cdot a\rangle\ge0$ ring‑wise on the **goal‑opposing** semicircle. Fine‑tune `W(θ)`, $\varepsilon_0$, and knee‑cap strength/width to close corridors.
- **Head‑on design** (`SignedPowerMagnetic.design_headon_tradeoff`) — compute minimal $r_2^{f}(v_n)$ for fixed $k_B$ and minimal $k_B^{f}(v_n)$ for fixed $r_2$ using measured $\eta$. Overlay validation trajectories.
- **APF potential & minima** (`plot_potential_and_minima`) — verify the repulsive wall budget and visualize candidate traps for APF.

---
### 7. Figure that can be generated <a name="figures"></a>

1) **APF vs Gyro comparison**
   - `figs/compare_trajectories_APF_vs_Magnetic.png` — trajectories with a **shared** speed colorbar. citeturn70search2
   - `figs/compare_speed_traces.png` — head‑on / off‑axis speed vs time. citeturn70search2
   - `figs/APF_potential_and_minima.png` — APF landscape with candidate minima. citeturn70search2

2) **Curvature & Grazing** (pick a magnetic law you want to showcase, e.g., `sine_rphase` or `tilted_sine`)
   - `figs/curv_maps_<law>.png` — curvature panels (total, B, geometric, goal). citeturn70search2
   - `figs/grazing_na_<law>.png` — grazing normal panels (total, B, geometric, goal). citeturn70search2

3) **Head‑on design**
   - `figs/design_headon_tradeoff.png` — $r_2^{f}(v_n)$ (fixed $k_B$) and $k_B^{f}(v_n)$ (fixed $r_2$) with optional overlays.

4) **Playful viz (optional, but popular)**
   - `figs/avoidance_side_by_side_multi.gif` — multi‑IC GIF with common time axis; APF background = $U_{\rm rep}$; Gyro background = signed curvature (black at zero); vortex strictly **inside** the obstacle. citeturn70search2


---

## References<a name="references"></a>

1. O. Khatib, “Real-Time Obstacle Avoidance for Manipulators and Mobile Robots,” *IJRR*, 1986.  
2. S. Paternain, D.E. Koditschek, A. Ribeiro, “Navigation Functions for Convex Potentials in a Space with Convex Obstacles,” *arXiv:1605.00638*, 2016.  
3. H. Kumar, S. Paternain, A. Ribeiro, “Navigation of a Quadratic Potential with Ellipsoidal Obstacles,” *arXiv:1908.08509*, 2022.  
4. A.D. Ames, X. Xu, J.W. Grizzle, P. Tabuada, “Control Barrier Function Based Quadratic Programs for Safety Critical Systems,” *IEEE TAC*, 2016.  
5. (Background) Texts on Riemannian/natural gradients and viability/Nagumo conditions.
6. O. Khatic, "The Potential Field Approach And Operational Space Formulation In Robot Control", In: Narendra, K.S. (eds) Adaptive and Learning Systems. Springer, Boston, MA, 1986.
7. Ratliff, N., Issa, J., & Kappler, D. (2020). Geometric Fabrics: Generalizing Classical Mechanics to Design Orbital Policies. -> "swirl" (gyroscopic term) defined within a Finsler-geometry framework/fundamentally the closest work to this.
8. Ratliff, N., et al. (2018). RMPflow: A Geometric Framework for Generation of Safe Motion. -> adaptation of M(q) metric
9. Huber, L., Billard, A., & Slotine, J. J. (2019). Avoidance of Convex and Concave Obstacles with Convergence Confirmation using Dynamical Systems. -> using mass matrix to alter dynamics, but doesn't take Levi-Civita identity into account.
10. Do Carmo, M. P.(2016). Differential Geometry of Curves and Surfaces: Revised and Updated Second Edition. Dover Books on Mathematics. -> Frenet/curvature (heading rate, signed curvature) 

