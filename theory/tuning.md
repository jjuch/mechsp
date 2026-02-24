# Tuning **M(q)** and **N(q)** for Safe and Agile Trajectory Shaping

> :warning: **This work is still under review and should not be used as a formal reference**

**Objective.** We study obstacle‑aware trajectory shaping **without** modifying the goal potential
$$
\psi(q)=\tfrac12\|q-q_g\|^2,
$$
so that the “heavy lifting” is accomplished by a **Riemannian metric** $M(q)$ and a **skew‑symmetric** field $N(q)$. We formalize the model from the **Lagrangian** up to the **Levi–Civita connection**, explain how the **natural‑gradient flow** and **gyroscopic (skew)** terms arise, derive **near‑obstacle asymptotics**, and end with **tuning rules** (with figures) that are consistent with **Nagumo’s invariance** condition. The figures validate the story and quantify recommended parameter choices.

> **Short executive summary.**
> - We keep $ \psi(q)=\tfrac12\|q-q_g\|^2 $ unchanged (no log‑barriers).  
> - The **metric** $ M(q) $ attenuates normal motion near obstacles; the **skew** term $ N(q)=-N(q)^\top $ bends trajectories but does no work on $ \psi $.  
> - With normal amplification $ \lambda_n(d)\sim (\sigma/d)^p (p>1) $, the natural‑gradient **normal component** scales as $ g^\natural_n\sim d^{p} $.  
> - **Nagumo invariance** at the obstacle requires $ n\cdot \dot q \ge 0 $. A **sufficient** and simple law is
>   $$
>   \boxed{\beta(d)=k\,d^{p}\quad (p>1),}
>   $$
>   optionally **tangential‑only** near the boundary (i.e., add $ \beta(d)|g^\natural_n|\,t $), which further protects invariance and bounds curvature.  
> - We provide a **ring‑based estimate** of a conservative $ k_{\text{safe}} $ and a 4‑step **tuning recipe**.  
> - Figures **A, B, C, D, F** validate these claims.

---

## 1. Lagrangian with metric $M(q)$, gauge 1‑form $A(q)$, two‑form $B=dA$, and viscous damping

We model the agent on the configuration space $\mathcal Q\subset\mathbb R^2$ via the Lagrangian

$$
\boxed{\;
\mathcal L(q,\dot q)=\tfrac12\,\dot q^\top M(q)\,\dot q \;+\; A(q)\!\cdot\!\dot q \;-\; \psi(q)\,,
\;}
$$

with $M(q)\succ0$ (Riemannian metric), $A(q)$ a gauge 1‑form, and **barrier‑free** $\psi(q)=\tfrac12\|q-q_g\|^2$. The **magnetic two‑form** is

$$
B(q)\;=\;dA(q)\,,\text{with } B_{ij}(q)=\partial_i A_j(q)-\partial_j A_i(q),\\ B(q)^\top=-B(q).
$$

Include a **Rayleigh damping** $$R(q,\dot q)=\tfrac12\,c\,\dot q^\top M(q)\dot q.$$ The Euler–Lagrange equations

$$
\frac{\mathrm d}{\mathrm dt}\big(\partial_{\dot q}\mathcal L\big)-\partial_q\mathcal L=\partial_{\dot q}R
$$

give

$$
\boxed{\;
M(q)\,\ddot q \;+\; C(q,\dot q)\,\dot q \;+\; c\,M(q)\,\dot q \;+\; \nabla\psi(q)
\;=\; B(q)\,\dot q\,,
\;}
$$

where $C(q,\dot q)\dot q$ comes from the **Levi–Civita connection** of $M$, componentwise:

$$
\big(C(q,\dot q)\dot q\big)^i=\sum_{j,k}\Gamma^i_{jk}(q)\,\dot q^j\dot q^k,\\
\Gamma^i_{jk}=\frac12\,\sum_\ell M^{i\ell}\Big(\partial_j M_{\ell k}+\partial_k M_{\ell j}-\partial_\ell M_{jk}\Big).
$$

Define the **total energy**

$$
V(q,\dot q):=\tfrac12\,\dot q^\top M(q)\,\dot q+\psi(q).
$$

Using the Levi–Civita identity $$\dot q^\top C(q,\dot q)\dot q=\tfrac12\,\dot q^\top \dot M(q)\,\dot q$$ and the skewness of $B$, we obtain the **exact energy balance** (no overdamped assumption):

$$
\boxed{\;
\dot V \;=\; -\,c\,\dot q^\top M(q)\,\dot q \;\le\; 0\,,\qquad
\text{(both \(C\) and \(B\) do no work on \(V\)).}
\;}
$$

#### 1.1 Skew map as a block rotation $N(q)$ in the $(t,n)$ frame
Let the **signed distance** to the nearest obstacle be $ d(q) $ (positive outside, zero on the boundary). At a boundary point, define the **outward unit normal** $ n(q) $, and let $ t(q)=J n(q) $ denote the in‑plane **tangent** (90° rotation). We use the orthonormal frame $ R=[\,t\;n\,].$

In 2‑D, every two‑form is a scalar field times the canonical rotation: $B(q)=b(q)\,J$. In the local orthonormal frame $R=[t\;\;n]$ attached to the obstacle boundary, write the velocity as $$v= v_t\,t+v_n\,n.$$ Then

$$
\begin{align}
B(q)\,v&=b(q)\,J\,v\\&=b(q)\,(v_t\,Jt+v_n\,Jn)\\&=b(q)\,(v_t\,n-v_n\,t).
\end{align}
$$

In the coordinates $(v_t,v_n)$, the action of $B$ is the **block rotation**

$$
\boxed{\;
N(q)\;\equiv\;\begin{bmatrix}0& -\,b(q)\\[2pt] b(q)& 0\end{bmatrix},
 \text{i.e.}\\
\begin{bmatrix} \dot v_t\\ \dot v_n \end{bmatrix}_{\!\!B\text{-part}}
\;=\;
\begin{bmatrix}0& -\,b(q)\\ b(q)& 0\end{bmatrix}
\begin{bmatrix} v_t \\ v_n \end{bmatrix}.
\;}
$$

This results in a pure gyroscopic connection $N(q)$: a **skew** coupling that bends $(v_t,v_n)$ without doing work.

> Relation to a **first‑order** bending term. In a **designed** first‑order field $$\dot q = -G^{-1}\nabla\psi + \beta(\cdot)J\,G^{-1}\nabla\psi,$$ the scalar $\beta(\cdot)$ plays the role of the **distance‑dependent** “magnetic” strength. In the full second‑order system above, the *implemented* gyroscopic strength is $b(q)$ in $B=bJ$. The two are related through the inertial/damping scales of the controller/plant; see the tuning note in §5.2.

---

## 2. Near‑obstacle asymptotics (second‑order)

Let the **inverse metric** be

$$
G^{-1}(q)=R\,\mathrm{diag}\!\big(1,\tfrac{1}{\lambda_n(d)}\big)\,R^\top,\\
\lambda_n(d)=1+\Big(\frac{\sigma}{d}\Big)^{p},\ \ p>1,
$$

so that the **natural gradient** satisfies, for bounded $\nabla\psi$,

$$
g^\natural_t=\langle G^{-1}\nabla\psi,t\rangle=\mathcal O(1),\\
g^\natural_n=\langle G^{-1}\nabla\psi,n\rangle\sim C\,d^{p}\quad(d\to0^+).
$$

The **first‑order** picture (used for analysis and in the figures) gives

$$
n\cdot\dot q \;=\; -\,g_n^\natural + \beta(d)\,g_t^\natural
\;\approx\; -\,C_1\,d^{p} + \beta(d)\,C_2\,.
$$

By **Nagumo’s invariance** condition, a **sufficient** choice is

$$
\boxed{\;\beta(d)=k\,d^{p}\,,\qquad p>1,\;}
$$

optionally **tangential‑only** (replace $J\,g^\natural$ by $|g^\natural_n|\,t$) to remove normal injection.

In the **second‑order** system, along the **normal** direction,

$$
n\cdot\ddot q \;=\; n\cdot M^{-1}\!\Big(B\dot q - C(q,\dot q)\dot q - c\,M\dot q - \nabla\psi\Big).
$$

Near $d=0$ with $v_n\to0$, we have:

*   $n\cdot\big(B\dot q\big)=b(d)\,(t\cdot\dot q)$, i.e., **gyroscopic normal injection** proportional to the **tangential speed** and to $b(d)$;
*   $n\cdot(C\dot q)=\mathcal O(\|\dot q\|^2)$ (quadratic in speed);
*   damping contributes $ -c\, n\cdot\dot q$ (scaled by local metric), and $\nabla\psi$ is bounded.

Thus, to **suppress** inward normal acceleration arbitrarily close to the boundary, it suffices to ensure $b(d)\to 0$ sufficiently fast. Choosing

$$
\boxed{\;b(d)=\bar k\,d^{p}\quad (p>1)\;}
$$

makes the gyroscopic normal term $b(d)\,(t\cdot\dot q)$ of order $\mathcal O(d^p)$, which vanishes faster than any bounded tangential speed can create a persistent inward pull. With the **energy identity** $$\dot V=-c\,\dot q^\top M\dot q\le 0$$ and the **metric** boundary layer ($g^\natural_n\sim d^p$), this preserves exactly the **same tuning law** as in the first‑order analysis.

**Conclusion.** Keeping **$C(q,\dot q)$** and **$B(q)$** in the **underdamped** equations does **not** change the **exponent $p$** nor the **distance law** for the skew intensity. Use $$\beta(d)=k\,d^{p}$$ in the first‑order design (figures), or $$b(d)=\bar k\,d^{p}$$ in the second‑order $B=bJ;$ the two constants $k,\bar k$ differ by implementation/damping scales (see §5.2).


---

## 4. Near‑obstacle asymptotics, curvature, and boundedness

- **Normal asymptotics.** Because $ g_n^\natural\sim d^p $, the boundary layer enforces **vanishing normal speed** as $ d\to 0 $. With $\beta(d)=k d^{p}$, the skew injection is of higher‑order smallness and does not override the metric shielding.

- **Curvature.** In many APF‑like designs the curvature of streamlines scales approximately like $ \kappa\sim \beta(d)/d $. Hence naive constant gains can make $ \kappa $ blow up near the boundary (the classical “swirl” issue). Choosing $ \beta(d)=k d^{\alpha} $ with $ \alpha\ge 1 $ bounds curvature as $ d\to 0 $; our choice $ \alpha=p>1 $ is therefore **curvature‑friendly** as well as **invariance‑friendly**.

> These ideas were foreshadowed in the APF literature (swirl near obstacles), and formalized in viability theory (boundary tangency). We leverage them without modifying the potential.

---

## 5. Practical tuning recipe

#### 5.1 Practical rules (first‑order figures)

1.  Pick **$p>1$** and $\sigma$ in $\lambda_n(d)=1+(\sigma/d)^p$.
2.  Use **$\beta(d)=k\,d^p$** (tangential‑only optional).
3.  Validate with **ring sampling** (Figs. A–D).
4.  Take **$k_{\text{safe}}$** from Fig. F and use $k\le k_{\text{safe}}$.

#### 5.2 Mapping to second‑order $$B(q)=b(q)\,J$$

If you **implement** gyroscopic bending in the **second‑order** model via $B=b(d)\,J$,

$$
M\ddot q + C(q,\dot q)\dot q + c M\dot q + \nabla\psi = B\,\dot q,
$$

choose

$$
\boxed{\;b(d)=\bar k\,d^{p},\quad p>1,}
$$

with $\bar k$ set so that the **effective** bend matches the first‑order $\beta(d)=k\,d^p$ at typical operating speeds (the mapping is linear in the relevant scale: $\beta \sim \alpha\,b$, where $\alpha$ depends on the controller/plant gain $c$ and on how you nondimensionalize). The **exponent $p$** and the **distance‑law** $d^p$ are **unchanged**.

> **Levi–Civita in simulation.**  
> • If you integrate **first‑order**, $\Gamma$ does not appear (you’re not integrating $\ddot q$).  
> • If you integrate **second‑order** (as above), include $C(q,\dot q)\dot q$. It **does not** alter the tuning exponents (it is quadratic in $\dot q$ and does no work on $V$); it may shift **constants** (e.g., the numerical $k_{\text{safe}}$) - the ring‑based procedure captures that.
5. **(Optional) Certified safety.** If a proof‑level guarantee is needed under model mismatch or numerical stepping, wrap the nominal field in a **CBF–QP** enforcing $ \dot h\ge -\alpha h $ with $ h=d(q) $.

---

## 6. Figures 

### A. First-order model

All figures are generated by [`tuning_firstOrder_figures.py`](scripts/tuning_firstOrder_figures.py).

- **Fig A — Invariance on rings.** Fraction of $n\cdot v<0$ points vs $d$ for several gains.  
  ![`scripts/figs/figA_invariance_rings.png`](scripts/figs/figA_invariance_rings.png)

- **Fig B — Trajectories.** Metric‑only, +magnetic (const), +magnetic $\beta\sim d^{p-1}$, +magnetic $\beta\sim d^{p}$, and **tangential‑only** $\beta\sim d^{p}$.  
  ![`scripts/figs/figB_trajectories_modes.png`](scripts/figs/figB_trajectories_modes.png)

- **Fig C — Ring‑averaged speeds.** Outward normal vs tangential speed vs $d$.  
  ![`scripts/figs/figC_ring_speeds.png`](scripts/figs/figC_ring_speeds.png)

- **Fig D — Curvature maps.** $|\kappa|$ for full vs tangential‑only $\beta\sim d^{p}$.  
  ![`scripts/figs/figD_curvature_dp.png`](scripts/figs/figD_curvature_dp.png)

- **Fig F — Safe‑gain chart.** $k_{\max}(d)$ and suggested $k_{\text{safe}}$.  
  ![`scripts/figs/figF_kmax_safe.png`](scripts/figs/figF_kmax_safe.png)

### B. Second-order model

All figures are generated by [`tuning_secondOrder_figures.py`](scripts/tuning_secondOrder_figures.py).

*   **S1 — Trajectories (2nd‑order)** for several initial states and four “magnetic” laws:  $b(d)\in\{\text{none},\ \text{const},\ d^{p-1},\ d^{p}\}$.
![`scripts/figs2/figS1_trajectories_second_order.png`](scripts/figs2/figS1_trajectories_second_order.png)
*   **S2 — Energy $V(t)$ decay**, confirming $\dot V\le0$ for all laws (gyroscopic & Levi–Civita do no work).
![`scripts/figs2/figS2_energy_decay.png`](scripts/figs2/figS2_energy_decay.png)
*   **S3 — Minimum distance** $ \min_t d(t)$ as a safety proxy for each law; **$d^p$** law avoids near‑grazing/collisions that appear with **const** or **$d^{p-1}$**.
![`scripts/figs2/figS3_min_distance.png`](scripts/figs2/figS3_min_distance.png)
*   **S4 — First‑ vs second‑order path** overlay (with $\beta(d)=k\,d^p$ vs $B=d^pJ$), showing **qualitatively similar** paths (same law).
![`scripts/figs2/figS4_first_vs_second.png`](scripts/figs2/figS4_first_vs_second.png)
*   **S5 — Safe‑gain chart (2nd‑order)**, obtained from the same **ring‑based** estimate $k_{\max}(d)=\max(g_n^\natural/(d^p g_t^\natural))$; we suggest a conservative $k_{\text{safe}}=\frac12\min_d k_{\max}(d)$.
![`scripts/figs2/figS5_kmax_safe_second_order.png`](scripts/figs2/figS5_kmax_safe_second_order.png)

> **Parameters used in the example.**  
> $p=2$, $\sigma=0.25$, damping $c=1.5$, obstacle: one disk of radius $0.5$ at the origin, goal $(1.2,1.1)$. We used $b(d)=\bar k\,d^p$ with $\bar k=1$.


---

## 7. Limitations and extensions

- **Continuous‑time vs numerics.** Nagumo invariance is a continuous‑time condition; explicit integrators and large steps can cause small penetrations. Use smaller steps near obstacles or a discrete‑time safety filter.

- **Speed normalization.** To compare “time to goal” across modes fairly, normalize speed away from the boundary layer; otherwise metric anisotropy also rescales speeds.

- **Second‑order (mechanical) systems.** In dynamics, skew (gyroscopic) terms still do no work; combine metric shaping with damping injection. For hard safety: CBF–QP atop the geometric field.

---

## References

1. O. Khatib, “Real-Time Obstacle Avoidance for Manipulators and Mobile Robots,” *IJRR*, 1986.  
2. S. Paternain, D.E. Koditschek, A. Ribeiro, “Navigation Functions for Convex Potentials in a Space with Convex Obstacles,” *arXiv:1605.00638*, 2016.  
3. H. Kumar, S. Paternain, A. Ribeiro, “Navigation of a Quadratic Potential with Ellipsoidal Obstacles,” *arXiv:1908.08509*, 2022.  
4. A.D. Ames, X. Xu, J.W. Grizzle, P. Tabuada, “Control Barrier Function Based Quadratic Programs for Safety Critical Systems,” *IEEE TAC*, 2016.  
5. (Background) Texts on Riemannian/natural gradients and viability/Nagumo conditions.
