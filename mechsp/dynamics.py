# mechsp/dynamics.py
import numpy as np
from .magnetics import grad_Bz, grad_Bz_analytic
from .barrier import edge_barrier_force

try:
    from scipy.integrate import solve_ivp
    _HAS_SCIPY_IVP = True
except Exception:
    _HAS_SCIPY_IVP = False

def force_from_currents(q, coil_xy, h, I, marble_moment=1.0, scale=1.0):
    """
    Vectorized across coils for a single point q: sum_i I_i * m * ∇Bz_i(q).
    """
    # broadcast xy (1,2) vs coils (N,2) inside grad_Bz_analytic by passing 1x2 np.array
    G_all = marble_moment * np.vstack([
        grad_Bz_analytic(q[None, :], cxy, h, scale=scale)[0] for cxy in coil_xy
    ])  # shape (N,2)
    return (I[:, None] * G_all).sum(axis=0)  # (2,)


def simulate_second_order(q0, v0, t_final, dt, force_fn, mass=1.0, damping=0.1,
                          goal=None, stop_tol=1e-3, speed_tol=5e-3):
    """
    Semi-implicit Euler for m qdd + c qd = F(q).
    Lightly damped: choose small damping (e.g., 0.05--0.2).
    Returns trajectory arrays (T, Q, V).
    """
    steps = int(np.ceil(t_final / dt))
    Q = np.zeros((steps + 1, 2))
    V = np.zeros((steps + 1, 2))
    T = np.linspace(0.0, steps * dt, steps + 1)
    q = q0.copy()
    v = v0.copy()
    Q[0], V[0] = q, v

    for k in range(steps):
        print(f"Simulate second order: step {k} of {steps}", end='\r')
        F = force_fn(q)
        # v_{k+1} = v_k + dt * (F - c v_k)/m
        v = v + dt * (F - damping * v) / mass
        # q_{k+1} = q_k + dt * v_{k+1}
        q = q + dt * v
        Q[k + 1], V[k + 1] = q, v

        if goal is not None:
            to_goal = np.linalg.norm(q - goal)
            if to_goal < stop_tol and np.linalg.norm(v) < speed_tol:
                T = T[:k + 2]
                Q = Q[:k + 2]
                V = V[:k + 2]
                break

    return T, Q, V


def simulate_second_order_ivp(q0, v0, t_final, force_interp, L,
                              mass=1.0, damping=0.1,
                              k_edge=5.0, margin=0.02, p_edge=2,
                              goal=None, stop_tol=1e-3, speed_tol=5e-3, max_step=0.01):
    """
    Integrate m qdd + c qd = F_interp(q) + F_edge(q).
    Uses SciPy's solve_ivp (LSODA) if available; else falls back to a manual semi-implicit loop.

    force_interp: callable q -> F_coils(q)
    """
    def rhs(t, y):
        q = y[0:2]
        v = y[2:4]
        F = force_interp(q) + edge_barrier_force(q, L, k_edge=k_edge, margin=margin, p=p_edge)
        a = (F - damping * v) / mass
        return np.hstack([v, a])

    y0 = np.hstack([q0, v0])

    if _HAS_SCIPY_IVP:
        # Stopping event near goal
        def event_goal(t, y):
            if goal is None:
                return 1.0
            q = y[0:2]
            v = y[2:4]
            return max(np.linalg.norm(q - goal) - stop_tol, np.linalg.norm(v) - speed_tol)
        event_goal.terminal = True
        event_goal.direction = -1.0

        sol = solve_ivp(rhs, (0.0, t_final), y0, method='LSODA',
                        max_step=max_step, rtol=1e-6, atol=1e-9,
                        events=event_goal)
        print("solve_ivp has converged...")
        T = sol.t
        Y = sol.y.T
        Q = Y[:, 0:2]
        V = Y[:, 2:4]
        return T, Q, V
    else:
        # fallback: manual loop (coarser)
        print("Manual solver loop")
        dt = max_step
        steps = int(np.ceil(t_final / dt))
        T = np.linspace(0.0, steps * dt, steps + 1)
        Q = np.zeros((steps + 1, 2))
        V = np.zeros((steps + 1, 2))
        q, v = q0.copy(), v0.copy()
        Q[0], V[0] = q, v
        for k in range(steps):
            print(f"Simulate second order: step {k} of {steps}", end='\r')
            F = force_interp(q) + edge_barrier_force(q, L, k_edge=k_edge, margin=margin, p=p_edge)
            v = v + dt * (F - damping * v) / mass
            q = q + dt * v
            Q[k+1], V[k+1] = q, v
            if goal is not None:
                if (np.linalg.norm(q - goal) < stop_tol) and (np.linalg.norm(v) < speed_tol):
                    T = T[:k+2]; Q = Q[:k+2]; V = V[:k+2]
                    break
        return T, Q, V
    

def simulate_second_order_chunked(
    q0, v0, t_final, force_fn, L,
    mass=1.0, damping=0.1,
    k_edge=5.0, margin=0.02, p_edge=2,
    goal=None, stop_tol=1e-3, speed_tol=5e-3,
    max_step=0.01,
    chunk_horizon=0.3,
    progress_cb=None,
):
    """
    Progress-aware simulator:
      - Integrates in time chunks of length 'chunk_horizon' (seconds)
      - After each chunk, calls progress_cb(progress in [0,1]) if provided.
      - Stops early if goal and low speed achieved.

    force_fn: callable q -> F_coils(q)
    """

    def rhs(t, y):
        q = y[0:2]
        v = y[2:4]
        F = force_fn(q) + edge_barrier_force(q, L, k_edge=k_edge, margin=margin, p=p_edge)
        a = (F - damping * v) / mass
        return np.hstack([v, a])

    # Accumulate solution
    T_list = [0.0]
    Q_list = [q0.copy()]
    V_list = [v0.copy()]
    y = np.hstack([q0, v0])

    t = 0.0
    # optionally report initial progress
    if progress_cb is not None:
        progress_cb(0.0)

    # helper for stop condition
    def reached(y_vec):
        if goal is None:
            return False
        q = y_vec[0:2]
        v = y_vec[2:4]
        return (np.linalg.norm(q - goal) < stop_tol) and (np.linalg.norm(v) < speed_tol)

    if reached(y):
        if progress_cb is not None:
            progress_cb(1.0)
        return np.array(T_list), np.array(Q_list), np.array(V_list)

    # Chunk loop
    while t < t_final - 1e-12:
        t_end = min(t + chunk_horizon, t_final)

        if _HAS_SCIPY_IVP:
            sol = solve_ivp(
                rhs, (t, t_end), y, method='LSODA',
                max_step=max_step, rtol=1e-6, atol=1e-9
            )
            # append the interior points (skip duplicate t)
            if sol.t.size > 1:
                T_list.extend(sol.t[1:].tolist())
                Q_list.extend(sol.y[0:2, 1:].T.tolist())
                V_list.extend(sol.y[2:4, 1:].T.tolist())
            # update state
            y = sol.y[:, -1]
            t = sol.t[-1]
        else:
            # fallback: manual semi-implicit loop over this chunk
            dt = max_step
            steps = int(np.ceil((t_end - t) / dt))
            for _ in range(steps):
                q = y[0:2]
                v = y[2:4]
                F = force_fn(q) + edge_barrier_force(q, L, k_edge=k_edge, margin=margin, p=p_edge)
                v = v + dt * (F - damping * v) / mass
                q = q + dt * v
                t = t + dt
                y = np.hstack([q, v])
                T_list.append(t)
                Q_list.append(q.tolist())
                V_list.append(v.tolist())

        # progress callback
        if progress_cb is not None:
            # print(f"Progressed: {min(1.0, t / t_final)}")
            progress_cb(min(1.0, t / t_final))

        # early stop check
        if reached(y):
            break

    return np.array(T_list), np.array(Q_list), np.array(V_list)