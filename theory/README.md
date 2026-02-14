# Introduction for Non-Experts**

## 1. Why move objects using magnetic fields?

This project creates a **programmable landscape of forces** on a flat surface.  
A small ball containing a magnetic particle rolls on that surface. Below the surface is a grid of coils—like an LED display, but producing magnetic fields instead of light. By changing the electrical current in each coil, you can change the magnetic field above it.

Because magnetic fields exert forces on the magnetic ball, the ball can be moved **without touching it**.

No motors.  
No sensors.  
No electronics inside the ball.  
The environment does all the work.

***

## 2. Why call this an “artificial mechanical spacetime”?

A rolling ball normally obeys Newton’s law:

$$
m\ddot q = F.
$$

But by shaping the magnetic environment, we can change:

*   the **potential landscape** (like hills and valleys the ball rolls on),
*   the **curvature of paths** (like adding built‑in sideways drift),
*   the **effective inertia** (making the ball feel “heavier” in some directions).

These effects resemble how real mechanical systems behave in curved spaces or under gyroscopic forces. In robotics and physics, this is called shaping an **artificial spacetime**.

***

## 3. What are $$K(q)$$, $$N_{\mathrm{eff}}(q)$$, and $$M_{\mathrm{eff}}(q)$$?

*   **$$K(q)$$**  
    A magnetic potential energy landscape created by static (DC) coil currents.  
    This determines *where* the ball wants to go.

*   **$$N_{\mathrm{eff}}(q)$$**  
    A gyroscopic-like effect created by rotating (AC) magnetic fields.  
    This determines *how the ball curves or swirls* as it moves.

*   **$$M_{\mathrm{eff}}(q)$$**  
    An effective mass matrix created by very fast oscillatory fields.  
    This determines *how the ball accelerates in different directions*.

By changing the patterns and timing of the coil currents, we can shape all three.

***

## 4. Why is this interesting?

Because:

*   the ball has **no intelligence**,
*   the environment does all the shaping,
*   the motion emerges from **physics**, not computation.

This lets you manipulate **many balls at once**, with guaranteed passivity and robustness, without needing tracking or feedback.

***