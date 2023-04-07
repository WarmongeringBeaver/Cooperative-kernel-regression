""" Utilities for the project."""


from math import ceil, sqrt

import numpy as np
import plotly.graph_objects as go

from .algs import run_FedAvg

### CONSTANTS
NB_POINTS_PLOT = 1000


def susbample_data(x, y, n=100, m: None | int = None, a=5, rng=None):
    """Returns a subsample of the complete data.

    Randomly selects `m` points from the first `n` data points of `x` and `y`,
    and distribute them evenly among `a` agents.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        The input data.
    y : array-like, shape (n_samples,)
        The target variable.
    n : int, optional (default=100)
        The number of first data points to consider from `x` and `y`.
    m : int or None, optional (default=None)
        The number of data points to select from `x` and `y`. If None, set to `ceil(sqrt(n))`.
    a : int, optional (default=5)
        The number of agents to distribute the selected data points to.
    rng : numpy.random.Generator or int or None, optional (default=None)
        A random number generator or seed to use for reproducibility.
        If None, use `np.random.default_rng()`.

    Returns
    -------
    n_n : array-like, shape (n,)
        The first `n` feature variables.
    y_n : array-like, shape (n,)
        The first `n` target variables.
    idx_sel_flat : array-like, shape (m,)
        The indices of the selected data points.
    idx_sel : array-like, shape (a, m // a)
        The indices of the selected data points attributed to each agent.
    x_sel_flat : array-like, shape (m, n_features)
        The selected `x` data points for all agents.
    x_sel : array-like, shape (a, m // a, n_features)
        The selected `x` data points attributed to each agent.
    y_sel_flat : array-like, shape (m,)
        The selected `y` data points for all agents.
    y_sel : array-like, shape (a, m // a)
        The selected `y` data points attributed to each agent.
    """
    if rng is None:
        rng = np.random.default_rng()
    if m is None:
        m: int = ceil(sqrt(n))
    assert (
        m % a == 0
    ), "The total number of points must be a multiple of the number of agents!"
    nb_points_per_agent: int = m // a

    print(f"n = {n}\nm = {m}\na = {a}\nnb_points_per_agent = {nb_points_per_agent}")
    msg: str = f"\nSelected {m} points to distribute to the {a} agents among the {n} first data points,"
    msg += f" resulting in {nb_points_per_agent} points per agent."
    print(msg)

    y_n = y[:n]
    x_n = x[:n]

    # select m points at random among the n first
    idx_sel_flat = rng.choice(range(n), m, replace=False)
    idx_sel = idx_sel_flat.reshape((a, nb_points_per_agent)).copy()

    # x_sel[k] holds the m // a "x" data points of agent k
    # while x_sel_flat holds the m data points of all agents
    x_sel_flat = np.array([x[i] for i in idx_sel_flat])
    x_sel = x_sel_flat.reshape((a, nb_points_per_agent))

    # idem for the y's
    y_sel_flat = np.array([y[i] for i in idx_sel_flat])
    y_sel = y_sel_flat.reshape((a, nb_points_per_agent))

    return x_n, y_n, idx_sel_flat, idx_sel, x_sel_flat, x_sel, y_sel_flat, y_sel


def build_kernel_matrices(x, x_sel_flat, n, m, idx_sel_flat):
    """Builds the kernel matrices.

    The matrices share the same memory.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        The complete array of data points.
    x_sel_flat : array-like, shape (m, n_features)
        The selected `x` data points for all agents.
    n : int, optional (default=100)
        The number of first data points to consider from `x` and `y`.
    m : int or None, optional (default=None)
        The number of data points to select from `x` and `y`. If None, set to `ceil(sqrt(n))`.
    idx_sel_flat : array-like, shape (m,)
        The indices of the selected data points.

    Returns
    -------
    K_nm : array-like, shape (n, m)
    K_mm : array-like, shape (m, m)

    """
    K_nm = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K_nm[i, j] = np.exp(-np.abs(x[i] - x_sel_flat[j]) ** 2)

    K_mm = K_nm[idx_sel_flat, :]  # shared memory

    return K_nm, K_mm


def plot_opt_gap_per_agent(
    a,
    t_max,
    alpha_opti,
    alpha_seq,
    x_log=True,
    y_log=True,
    y_range=None,
) -> None:
    """Plots the optimality gap for each agent."""
    fig = go.Figure()
    # maximum NB_POINTS_PLOT points for plotly plotting
    if t_max > NB_POINTS_PLOT:
        print(
            f"Warning: t_max > {NB_POINTS_PLOT}, plotting only {NB_POINTS_PLOT} points."
        )
    plot_samples = range(1, t_max + 1, t_max // NB_POINTS_PLOT + 1)
    for agent in range(a):
        # add trace for agent
        fig.add_trace(
            go.Scatter(
                x=list(plot_samples),
                y=[
                    np.linalg.norm(alpha_opti - alpha_seq[t][agent])
                    for t in plot_samples
                ],
                name=f"Agent {agent+1}",
                mode="lines+markers",
            )
        )
    fig.update_layout(
        title="Optimality gap for each agent",
        xaxis_title="Iteration" + (" (log)" if x_log else ""),
        yaxis_title="Optimality gap" + (" (log)" if y_log else ""),
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    fig.show()


def plot_opt_gap(t_max, alpha_opti, alpha_seq, x_log=True, y_log=True) -> None:
    """Plots the optimality gap for a single sequence of alpha's."""
    fig = go.Figure()
    # maximum NB_POINTS_PLOT points for plotly plotting
    if t_max > NB_POINTS_PLOT:
        print(
            f"Warning: t_max > {NB_POINTS_PLOT}, plotting only {NB_POINTS_PLOT} points."
        )
    plot_samples = range(1, t_max + 1, t_max // NB_POINTS_PLOT + 1)
    # plot optimality gap
    fig.add_trace(
        go.Scatter(
            x=list(plot_samples),
            y=[np.linalg.norm(alpha_opti - alpha_seq[t]) for t in plot_samples],
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="Optimality gap",
        xaxis_title="Iteration" + (" (log)" if x_log else ""),
        yaxis_title="Optimality gap" + (" (log)" if y_log else ""),
    )
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    fig.show()


def f(x_prime: float, alpha: np.ndarray, x_n: np.ndarray) -> float:
    """Computes the function f(x) for a given alpha."""
    k = np.exp(-np.abs(x_prime - x_n) ** 2)
    return np.sum(alpha * k)


def plot_f(
    n, x_n, y_n, x_prime, alpha_opti, alpha_agents: np.ndarray | None, x_sel_flat
):
    """Plots the obtained function for optimal alpha along training data."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_n,
            y=y_n,
            name=f"{n} first samples of y vs x",
            mode="markers",
            marker=dict(color="gray", size=5, opacity=0.5),
        )
    )
    if alpha_agents is not None:
        a = alpha_agents.shape[0]
        for agent in range(a):
            fig.add_trace(
                go.Scatter(
                    x=x_prime,
                    y=[f(x_p, alpha_agents[agent], x_sel_flat) for x_p in x_prime],
                    name="f(x) for final alpha" + (a > 1) * f" of agent {agent+1}",
                    mode="lines",
                )
            )
    fig.add_trace(
        go.Scatter(
            x=x_prime,
            y=[f(x_p, alpha_opti, x_sel_flat) for x_p in x_prime],
            name="f(x) for optimal alpha",
            mode="lines",
            line=dict(color="teal", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Obtained function for optimal alpha and fit to data",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        width=1200,
        height=600,
    )

    fig.show()


def print_vector_norms(a, t_max, vec_seq, name: str, x_log=True, y_log=True) -> None:
    """Plots the optimality gap for each agent."""
    fig = go.Figure()
    skip_0 = 0
    if x_log:
        print("Skipping first iteration for x log plot.")
        skip_0 = 1
    # maximum NB_POINTS_PLOT points for plotly plotting
    if t_max > NB_POINTS_PLOT:
        print(
            f"Warning: t_max > {NB_POINTS_PLOT}, plotting only {NB_POINTS_PLOT} points."
        )
    plot_samples = range(skip_0, t_max + skip_0, t_max // NB_POINTS_PLOT + 1)
    for agent in range(a):
        # add trace for agent
        fig.add_trace(
            go.Scatter(
                x=list(plot_samples),
                y=[np.linalg.norm(vec_seq[t][agent]) for t in plot_samples],
                name=f"Agent {agent+1}",
                mode="lines+markers",
            )
        )
    fig.update_layout(
        title=f"||{name}|| for each agent",
        xaxis_title="Iteration" + (" (log)" if x_log else ""),
        yaxis_title=f"||{name}||" + (" (log)" if y_log else ""),
    )
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    fig.show()


def show_selected_points(x_n, y_n, n, x_sel_flat, y_sel_flat, m, a) -> None:
    """Shows the randomly selected points."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_n, y=y_n, mode="markers", name=f"{n} first data points")
    )
    fig.add_trace(
        go.Scatter(
            x=x_sel_flat,
            y=y_sel_flat,
            mode="markers",
            name=f"{m} randomly selected samples",
        )
    )
    fig.update_layout(
        title=f"Selected samples to distribute among {a} agents",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )
    fig.show()


def check_W(W) -> None:
    """Checks for symmetry and double stochasticity."""
    assert (W.T == W).all(), "W is not symmetric!"
    col_wise = np.sum(W, axis=0)
    row_wise = np.sum(W, axis=1)
    assert (col_wise == 1).all() and (
        row_wise == 1
    ).all(), "W is not double stochastic!"


def build_A(W, a):
    """Builds the A matrix for the Dual Decomposition formulation."""
    # the first dim of A is unknown a priori,
    # so we build a list of rows
    list_rows = []
    assert (a, a) == W.shape
    for k in range(a):
        # in the slides p < k, but k < p is more natural to me ^^
        for p in range(k + 1, a):
            if W[k, p] != 0:  # k ~ p
                row = np.zeros((a))
                row[k], row[p] = 1, -1
                list_rows.append(row)
    A = np.array(list_rows)
    return A


def build_kernel_matrices_out_of_dataset(
    x_m_points: np.ndarray, x: np.ndarray, n: int, a: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the kernel matrices for points not necessarily included in the dataset.

    Parameters
    ----------
    x_m_points : array-like, shape (m,)
        The (potentially) out-of-database data points.
    x : array-like, shape (number_of_agents, data_size_per_agent,)
        The points in the data.

    Returns
    -------
    K_nm : array-like, shape (n, m)
    K_mm : array-like, shape (m, m)
    K_dm_per_agent: array-like, shape (a, d, m)
    """
    # checks
    assert n % a == 0, "n is not a multiple of a!"
    # K_mm
    m = x_m_points.size
    K_mm = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K_mm[i, j] = np.exp(-np.abs(x_m_points[i] - x_m_points[j]) ** 2)
    # K_nm
    K_nm = np.zeros((n, m))
    x_flat = x.flatten()
    for i in range(n):
        for j in range(m):
            K_nm[i, j] = np.exp(-np.abs(x_flat[i] - x_m_points[j]) ** 2)
    # K_dm_per_agent
    d = n // a
    K_dm_per_agent = np.zeros((a, d, m))
    for k in range(a):
        for i in range(d):
            for j in range(m):
                K_dm_per_agent[k, i, j] = np.exp(-np.abs(x[k][i] - x_m_points[j]) ** 2)

    return K_nm, K_mm, K_dm_per_agent


def study_FedAvg(
    alpha_server_0,
    t_max,
    rng,
    a,
    c,
    m,
    K_dm_per_agent,
    K_mm,
    E,
    x,
    B,
    y,
    sigma,
    step_size_seq,
    alpha_opti,
    n,
    y_flat,
    x_prime,
    x_m_points,
) -> None:

    # run
    alpha_seq = run_FedAvg(
        alpha_server_0,
        t_max,
        rng,
        a,
        c,
        m,
        K_dm_per_agent,
        K_mm,
        E,
        x,
        B,
        y,
        sigma,
        step_size_seq,
    )

    # plots
    plot_opt_gap(t_max, alpha_opti, alpha_seq)
    plot_f(
        n,
        x.flatten(),
        y_flat,
        x_prime,
        alpha_opti,
        alpha_seq[-1].reshape((1, m)),
        x_m_points,
    )
