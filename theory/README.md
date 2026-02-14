# Artificial Mechanical Spacetimes — High‑Level Overview

This subfolder contains the theoretical foundations for **artificial mechanical spacetimes** created by magnetic field shaping using an array of programmable coils.

The ball manipulated in this environment contains a small magnetic dipole.  
The external environment (the coil array) generates a **multi‑frequency magnetic field** that induces:

- a **potential field** $K(q)$ through static (DC) magnetic gradients,
- a **gyroscopic connection** $N_{\mathrm{eff}}(q)$ through rotating (AC) fields and dipole phase lag,
- a **position‑dependent effective inertia** $M_{\mathrm{eff}}(q)$ through high‑frequency amplitude modulation (vibrational averaging).

Together, these terms yield effective second‑order mechanical dynamics:

$$
M_{\mathrm{eff}}(q)\ddot q + N_{\mathrm{eff}}(q)\dot q + \nabla K(q) = 0.
$$

This produces a **geometrically engineered mechanical spacetime** for the ball.

---

# Introduction for Non‑Experts

## 1. What the hardware looks like
Imagine a flat board with a grid of electromagnets underneath.  
On top of the board sits a small magnetic ball (e.g., a steel sphere or a sphere containing a permanent magnet).

By adjusting the currents in the electromagnets, one can **shape a magnetic field** over the surface.  
This magnetic field creates forces that move the ball **without touching it**.

---

## 2. Why call this an “artificial mechanical spacetime”?

Normally, a rolling ball obeys Newton’s law:

$$
m \ddot{q} = F(q, \dot{q}).
$$

But if we shape the magnetic field carefully, the ball behaves *as if*:

- the landscape has invisible hills and valleys (potential $`K(q)`$),
- the space has built‑in sideways drift (gyroscopic term $`N_{\mathrm{eff}}(q)`$),
- the ball becomes “heavier” in some directions than others (effective inertia $`M_{\mathrm{eff}}(q)`$).

These are all classical mechanical effects, but here they are **created artificially** by the magnetic environment.

This is why the system is called an *artificial mechanical spacetime*.

---

## 3. Why is this useful?

- The ball has **no electronics** or motors.
- The environment acts as the **controller**.
- Multiple balls can be manipulated **simultaneously**.
- The global field is robust and passive, so no high‑bandwidth feedback is needed.

---

## 4. What is in this folder?

### [`theory.md`](theory.md)
A fully rigorous, mathematically complete derivation of:

- the multi‑frequency electromagnetic field model,
- the static potential $K_{\mathrm{eff}}$,
- the gyroscopic connection $N_{\mathrm{eff}}$,
- the effective inertia correction $M_{\mathrm{eff}}$,
- all assumptions and cross‑coupling conditions,
- and implementation implications.

### [`references.md`](references.md)
Structured scientific citations.
