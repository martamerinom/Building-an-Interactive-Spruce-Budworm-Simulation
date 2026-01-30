import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def spruce_budworm(t: float, x: float, r: float = 0.5, k: float = 10) -> float:
    """This function returns the spruce budworm differential equation."""
    dxdt = r * x * (1 - (x / k)) - (x**2) / (1 + x**2)
    return dxdt


# (You have two functions with the same name. Keep ONE.
# We'll keep the one that marks equilibria, so you can delete/ignore the earlier one.)

def find_equilibria(r: float = 0.5, k: float = 10, ngrid: int = 2000):
    xg = np.linspace(0, k, ngrid)
    fg = r * xg * (1 - xg / k) - (xg**2) / (1 + xg**2)

    roots = []
    for i in range(len(xg) - 1):
        if fg[i] * fg[i + 1] < 0:
            root = brentq(
                lambda x: r * x * (1 - x / k) - (x**2) / (1 + x**2),
                xg[i],
                xg[i + 1]
            )
            roots.append(root)

    return roots


def stability(xeq, r=0.5, k=10, eps=1e-3):
    f_left = r * (xeq - eps) * (1 - (xeq - eps) / k) - ((xeq - eps)**2) / (1 + (xeq - eps)**2)
    f_right = r * (xeq + eps) * (1 - (xeq + eps) / k) - ((xeq + eps)**2) / (1 + (xeq + eps)**2)

    if f_left > 0 and f_right < 0:
        return "stable"
    elif f_left < 0 and f_right > 0:
        return "unstable"
    else:
        return "neutral"


def plot_spruce_budworm_rate(xt: int, r: float = 0.5, k: float = 10):
    x = np.linspace(0, k, xt)
    dxdt = r * x * (1 - x / k) - (x**2) / (1 + x**2)

    eq_points = find_equilibria(r=r, k=k)

    plt.figure()
    plt.plot(x, dxdt)
    plt.axhline(0)

    for xeq in eq_points:
        s = stability(xeq, r, k)
        if s == "stable":
            plt.plot(xeq, 0, "go")  # green dot
        elif s == "unstable":
            plt.plot(xeq, 0, "ro")  # red dot

    plt.xlabel("x (Budworm population)")
    plt.ylabel("dx/dt")
    plt.title("Spruce Budworm Growth Rate vs Population")
    plt.show()


def evolve_spruce_budworm(t: np.ndarray, x: np.ndarray, r: float = 0.5, k: float = 10, t_eval: float = 10):
    """
    Function that evolves the system forward in time using numerical integration.
    This solves the ODE and appends the results to existing time and population arrays.
    """

    # Define time span from last time point
    t_span = (t[-1], t[-1] + t_eval)

    # Create evaluation points along the time span
    t_eval_points = np.linspace(t_span[0], t_span[1], 100)

    # Solve the ODE
    solution = solve_ivp(
        fun=spruce_budworm,
        t_span=t_span,
        y0=[x[-1]],
        t_eval=t_eval_points,
        args=(r, k),
        method="RK45"
    )

    t_new = solution.t
    x_new = solution.y[0]

    # Ensure non-negative population
    x_new = np.clip(x_new, 0.0, None)

    # Avoid duplicating the last point when concatenating
    if t_new.size > 0 and np.isclose(t_new[0], t[-1]):
        t_new = t_new[1:]
        x_new = x_new[1:]

    # Concatenate results
    t = np.concatenate((t, t_new))
    x = np.concatenate((x, x_new))

    return t, x


# -----------------------
# PART 5: Time series visualization
# -----------------------
def plot_spruce_budworm(t: np.ndarray, x: np.ndarray):
    """
    Plot the budworm population dynamics over time.

    - Time on x-axis
    - Population on y-axis
    - Green trajectory
    - y-axis starts at 0
    - Returns fig, ax (Streamlit-friendly)
    """
    t = np.asarray(t)
    x = np.asarray(x)

    if t.size != x.size:
        raise ValueError("t and x must have the same length")

    x = np.clip(x, 0.0, None)

    fig, ax = plt.subplots()
    ax.plot(t, x, color="green")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Budworm population")
    ax.set_title("Spruce Budworm Population Dynamics")
    ax.grid(True)

    return fig, ax


# --- Optional quick test (uncomment to test locally) ---
if __name__ == "__main__":
     t = np.array([0.0])
     x = np.array([1.0])
     t, x = evolve_spruce_budworm(t, x, r=0.5, k=10, t_eval=10)
     fig, ax = plot_spruce_budworm(t, x)
     plt.show()
