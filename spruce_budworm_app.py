import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import your functions from SESSION_1.py
from SESSION_1_SECOND_ATTEMPT import (
    spruce_budworm,
    plot_spruce_budworm_rate,
    evolve_spruce_budworm,
    plot_spruce_budworm
)


st.title("Spruce Budworm Population Dynamics")

# Sidebar controls
r = st.sidebar.slider("Intrinsic growth rate (r)", 0.0, 1.0, 0.5, 0.01)
k = st.sidebar.slider("Carrying capacity (k)", 0.1, 10.0, 10.0, 0.1)

# App sets initial population to k/10 by default
x0 = k/10

t_eval = st.sidebar.slider("Time to evolve", 1, 100, 10, 1)
evolve_button = st.sidebar.button("Evolve Forward")

reset_button = st.sidebar.button("Reset simulation")

# Initialize session state
if "sbw_t" not in st.session_state or "sbw_x" not in st.session_state:
    st.session_state["sbw_t"] = np.array([0.0])
    st.session_state["sbw_x"] = np.array([x0])

# Reset if requested (also resets if k changes a lot and you want)
if reset_button:
    st.session_state["sbw_t"] = np.array([0.0])
    st.session_state["sbw_x"] = np.array([x0])

# Retrieve session data
t = st.session_state["sbw_t"]
x = st.session_state["sbw_x"]

# Evolve if requested
if evolve_button:
    t, x = evolve_spruce_budworm(t, x, r=r, k=k, t_eval=float(t_eval))
    st.session_state["sbw_t"] = t
    st.session_state["sbw_x"] = x

# Show the equation with current parameters
st.latex(r"\frac{dx}{dt} = r x\left(1-\frac{x}{k}\right) - \frac{x^2}{1+x^2}")
st.write(f"Current parameters: r = {r:.2f}, k = {k:.2f}")

# Plot phase portrait
st.subheader("Phase Portrait (dx/dt vs x)")
plot_spruce_budworm_rate(xt=400, r=r, k=k)  # this currently calls plt.show() in your code
st.pyplot(plt.gcf())
plt.clf()

# Plot time series
st.subheader("Time Series (x vs t)")
fig2, ax2 = plot_spruce_budworm(t, x)
st.pyplot(fig2)
