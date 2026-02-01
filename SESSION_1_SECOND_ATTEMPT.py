import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# -----------------------
# Model
# -----------------------
def spruce_budworm(t: float, x: float, r: float = 0.5, k: float = 10.0) -> float:
    """Spruce budworm ODE: dx/dt = f(x)."""
    return r * x * (1 - x / k) - (x**2) / (1 + x**2)


def f_rhs(x: float, r: float = 0.5, k: float = 10.0) -> float:
    """RHS f(x) = dx/dt (same as spruce_budworm but without t)."""
    return r * x * (1 - x / k) - (x**2) / (1 + x**2)


def fprime(x: float, r: float = 0.5, k: float = 10.0) -> float:
    """Derivative f'(x) for stability + critical slowing down rate."""
    # f(x) = r x - (r/k)x^2 - x^2/(1+x^2)
    # d/dx = (r - 2r x/k) - 2x/(1+x^2)^2
    return (r - 2 * r * x / k) - (2 * x) / (1 + x**2) ** 2


# -----------------------
# Equilibria + stability
# -----------------------
def find_equilibria(r: float = 0.5, k: float = 10.0, ngrid: int = 4000) -> list[float]:
    """
    Find equilibria in [0, k] by scanning for sign changes and using brentq.
    Also includes exact roots if a grid point hits f(x)=0.
    """
    xg = np.linspace(0.0, k, ngrid)
    fg = f_rhs(xg, r=r, k=k)

    roots = []

    # Add exact hits (rare but possible)
    tol = 1e-10
    for xi, fi in zip(xg, fg):
        if abs(fi) < tol:
            roots.append(float(xi))

    # Add roots from sign changes
    for i in range(len(xg) - 1):
        if fg[i] * fg[i + 1] < 0:
            root = brentq(lambda x: f_rhs(x, r=r, k=k), xg[i], xg[i + 1])
            roots.append(float(root))

    # Always include x=0 if it is a root (it is for this model)
    if abs(f_rhs(0.0, r=r, k=k)) < 1e-12:
        roots.append(0.0)

    # Clean + sort unique
    roots = sorted(set(round(rt, 10) for rt in roots))
    return [float(rt) for rt in roots]


def classify_equilibria(r: float = 0.5, k: float = 10.0):
    """Return list of (xeq, 'stable'/'unstable'/'neutral') using f'(xeq)."""
    eqs = find_equilibria(r=r, k=k)
    out = []
    for xeq in eqs:
        fp = fprime(xeq, r=r, k=k)
        if fp < 0:
            out.append((xeq, "stable"))
        elif fp > 0:
            out.append((xeq, "unstable"))
        else:
            out.append((xeq, "neutral"))
    return out


# -----------------------
# Numerical evolution
# -----------------------
def evolve_spruce_budworm(
    t: np.ndarray,
    x: np.ndarray,
    r: float = 0.5,
    k: float = 10.0,
    t_eval: float = 10.0,
):
    """
    Evolve forward by t_eval time units from the last (t, x).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    t_span = (t[-1], t[-1] + float(t_eval))
    t_eval_points = np.linspace(t_span[0], t_span[1], 200)

    def rhs(ti, yi):
        return spruce_budworm(ti, yi[0], r, k)

    sol = solve_ivp(rhs, t_span=t_span, y0=[x[-1]], t_eval=t_eval_points, method="RK45")

    t_new = sol.t
    x_new = sol.y[0]

    x_new = np.clip(x_new, 0.0, None)

    # avoid duplicating the first point
    if t_new.size > 0 and np.isclose(t_new[0], t[-1]):
        t_new = t_new[1:]
        x_new = x_new[1:]

    t_out = np.concatenate((t, t_new))
    x_out = np.concatenate((x, x_new))
    return t_out, x_out


def plot_spruce_budworm(t: np.ndarray, x: np.ndarray):
    t = np.asarray(t)
    x = np.asarray(x)

    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Budworm population")
    ax.set_title("Spruce Budworm Population Dynamics")
    ax.grid(True)
    return fig, ax


# -----------------------
# Q5: Critical slowing down measurements
# -----------------------
def time_to_escape_from_equilibrium(
    xeq: float,
    r: float = 0.5,
    k: float = 10.0,
    eps: float = 1e-3,
    Delta: float = 0.2,
    tmax: float = 500.0,
):
    """
    Start at x0 = xeq + eps and return the first time when |x(t) - xeq| >= Delta.
    If not reached within tmax, returns None.
    """
    x0 = xeq + eps

    def rhs(t, y):
        return [spruce_budworm(t, y[0], r, k)]

    def leave_event(t, y):
        return abs(y[0] - xeq) - Delta

    leave_event.terminal = True
    leave_event.direction = 1

    sol = solve_ivp(rhs, (0.0, tmax), [x0], events=leave_event, max_step=0.1)

    if len(sol.t_events[0]) == 0:
        return None
    return float(sol.t_events[0][0])


def time_to_change_by_amount(
    xstart: float,
    r: float = 0.5,
    k: float = 10.0,
    Delta: float = 1.0,
    tmax: float = 500.0,
):
    """
    Start at x0=xstart and return the first time when |x(t) - xstart| >= Delta.
    If not reached within tmax, returns None.
    """
    def rhs(t, y):
        return [spruce_budworm(t, y[0], r, k)]

    def change_event(t, y):
        return abs(y[0] - xstart) - Delta

    change_event.terminal = True
    change_event.direction = 1

    sol = solve_ivp(rhs, (0.0, tmax), [xstart], events=change_event, max_step=0.1)

    if len(sol.t_events[0]) == 0:
        return None
    return float(sol.t_events[0][0])


def plot_spruce_budworm_rate(xt: int = 400, r: float = 0.5, k: float = 10.0):
    """
    Phase portrait: plot dx/dt = f(x) versus x, marking equilibria.
    Returns (fig, ax) and does NOT call plt.show() (better for Streamlit).
    """
    x = np.linspace(0.0, k, int(xt))
    dxdt = f_rhs(x, r=r, k=k)

    fig, ax = plt.subplots()
    ax.plot(x, dxdt)
    ax.axhline(0.0, linewidth=1)

    # Mark equilibria (green = stable, red = unstable)
    eqs = classify_equilibria(r=r, k=k)
    for xeq, st in eqs:
        if 0.0 <= xeq <= k:
            ax.plot([xeq], [0.0], "o")
            # color by stability
            if st == "stable":
                ax.lines[-1].set_color("green")
            elif st == "unstable":
                ax.lines[-1].set_color("red")
            else:
                ax.lines[-1].set_color("gray")

    ax.set_xlabel("x (Budworm population)")
    ax.set_ylabel("dx/dt")
    ax.set_title("Spruce Budworm Growth Rate vs Population")
    ax.grid(True)
    return fig, ax


# -----------------------
# MAIN (only one!)
# -----------------------
if __name__ == "__main__":
    r, k = 0.5, 10.0

    # 1) Equilibria + stability
    eqs = classify_equilibria(r=r, k=k)
    print("Equilibria (x*, stability):")
    for xeq, st in eqs:
        print(f"  x* = {xeq:.6f} -> {st}, f'(x*) = {fprime(xeq, r, k):.6f}")

    unstable = [xeq for xeq, st in eqs if st == "unstable"]
    if not unstable:
        raise RuntimeError("No unstable equilibrium found — check parameters.")

    # Usually the middle one is unstable; if multiple, pick the middle by sorting:
    unstable = sorted(unstable)
    xu = unstable[len(unstable) // 2]
    lam = fprime(xu, r, k)

    # 2) Time-to-escape near unstable equilibrium
    eps = 1e-3
    Delta = 1.0

    T_near = time_to_escape_from_equilibrium(xu, r=r, k=k, eps=eps, Delta=Delta, tmax=500.0)
    T_theory = (1.0 / lam) * np.log(Delta / eps)

    # 3) Compare far from equilibrium
    T_far_1 = time_to_change_by_amount(1.0, r=r, k=k, Delta=Delta, tmax=500.0)
    T_far_8 = time_to_change_by_amount(8.0, r=r, k=k, Delta=Delta, tmax=500.0)

    print("\nCritical slowing down (Q5):")
    print(f"  Unstable equilibrium xu = {xu:.6f}")
    print(f"  Local growth rate lambda = f'(xu) = {lam:.6f}")
    print(f"  Measured escape time from xu+eps (eps={eps}) to |x-xu|>=Delta (Delta={Delta}): {T_near}")
    print(f"  Linear prediction T ≈ (1/lambda) ln(Delta/eps) = {T_theory:.4f}")

    print("\nFar-from-equilibrium comparison (same Delta):")
    print(f"  From x0=1: time to change by Delta={Delta}: {T_far_1}")
    print(f"  From x0=8: time to change by Delta={Delta}: {T_far_8}")

    print("\nInstantaneous speeds |dx/dt|:")
    print(f"  Near xu+eps: |f(xu+eps)| = {abs(f_rhs(xu + eps, r, k)):.6e}")
    print(f"  At x=1:      |f(1)|      = {abs(f_rhs(1.0, r, k)):.6e}")
    print(f"  At x=8:      |f(8)|      = {abs(f_rhs(8.0, r, k)):.6e}")

    # 4) Plot that ACTUALLY shows critical slowing down: start near xu, run long enough
    t = np.array([0.0])
    x = np.array([xu + eps])
    t, x = evolve_spruce_budworm(t, x, r=r, k=k, t_eval=120.0)  # long horizon
    plot_spruce_budworm(t, x)
    plt.show()


