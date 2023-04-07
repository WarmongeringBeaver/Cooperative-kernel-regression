"""Optimization algorithms."""

import numpy as np
from tqdm.auto import trange


def run_DGD(
    alpha_agents_0, t_max, a, K_mm, y_n, idx_sel, K_nm, sigma, m, W, step_size
) -> list[np.ndarray]:
    """Decentralized Gradient Descent algorithm."""
    alpha_seq = [alpha_agents_0]
    # alpha_agents will hold the current value of the alpha vectors
    alpha_agents = alpha_agents_0.copy()

    for _ in trange(t_max):
        # In DGD each agent computes its own gradient
        for k in range(a):
            grad_left = 1 / a * K_mm @ alpha_agents[k]
            y_agent_k = y_n[idx_sel[k]]
            K_agent_k = K_nm[idx_sel[k]]
            grad_right = (
                1 / sigma**2 * K_agent_k.T @ (y_agent_k - K_agent_k @ alpha_agents[k])
            )
            grad = grad_left - grad_right
            # compute mixing of alphas
            alphas_mixing = np.zeros((m))
            for j in range(a):  # loop over agents
                alphas_mixing += W[k, j] * alpha_agents[j]
            alpha_agents[k] = alphas_mixing - step_size * grad
        # store iteration
        alpha_seq.append(alpha_agents.copy())

    return alpha_seq


def run_GT(
    alpha_agents_0: np.ndarray,
    grad_agents_0: np.ndarray,
    t_max: int,
    a: int,
    K_mm: np.ndarray,
    y_n: np.ndarray,
    idx_sel: np.ndarray,
    K_nm: np.ndarray,
    sigma: float,
    m: int,
    W: np.ndarray,
    step_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Gradient Tracking algorithm."""
    alpha_seq: list[np.ndarray] = [alpha_agents_0]
    grad_seq: list[np.ndarray] = [grad_agents_0]
    # alpha{,grad}_agents will hold the current value of the alpha vectors
    alpha_agents = alpha_agents_0.copy()
    grad_agents = grad_agents_0.copy()

    for _ in trange(t_max):
        # alpha update (same as DGD)
        for k in range(a):
            # compute mixing of alphas
            alphas_mixing = np.zeros((m))
            for j in range(a):  # loop over agents
                alphas_mixing += W[k, j] * alpha_agents[j]
            alpha_agents[k] = alphas_mixing - step_size * grad_agents[k]
        # store alpha iteration
        alpha_seq.append(alpha_agents.copy())
        # gradient update with new (and old) alpha
        # (at this point `grad_agents` still holds the gradients on the old alphas
        # but `alpha_agents` has been updated)
        for k in range(a):
            grad_left = 1 / a * K_mm @ alpha_agents[k]
            y_agent_k = y_n[idx_sel[k]]
            K_agent_k = K_nm[idx_sel[k]]
            grad_right = (
                1 / sigma**2 * K_agent_k.T @ (y_agent_k - K_agent_k @ alpha_agents[k])
            )
            new_grad = grad_left - grad_right
            grad_diff = new_grad - grad_agents[k]
            # compute mixing of gradients
            grads_mixing = np.zeros((m))
            for j in range(a):  # loop over agents
                grads_mixing += W[k, j] * grad_agents[j]
            grad_agents[k] = grads_mixing + grad_diff
        # store gradient iteration
        grad_seq.append(grad_agents.copy())

    return alpha_seq, grad_seq


def run_DD(
    alpha_agents_0: np.ndarray,
    lambda_agents_0: np.ndarray,
    t_max: int,
    a: int,
    K_mm: np.ndarray,
    y_n: np.ndarray,
    idx_sel: np.ndarray,
    K_nm: np.ndarray,
    sigma: float,
    m: int,
    W: np.ndarray,
    step_size: float,
    beta_penalize: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Dual decomposition algorithm.

    Local updates are done with exact gradient solving.
    """
    # checks
    assert alpha_agents_0.shape == (a, m)
    assert lambda_agents_0.shape == (a, a, m)

    alpha_seq: list[np.ndarray] = [alpha_agents_0]
    lambda_seq: list[np.ndarray] = [lambda_agents_0]
    # alpha{,lambda}_agents will hold the current value of the alpha vectors
    alpha_agents = alpha_agents_0.copy()
    lambda_agents = lambda_agents_0.copy()

    for _ in trange(t_max):
        ### local α update (exact gradient solving)
        # -> solve H * α_k = b to find α_k
        # s.t. ∇L_k(α_k, lambda_k) = 0
        for k in range(a):
            # H = 1 / a * K_mm + 1 / σ**2 * K_agent_k.T @ K_agent_k
            H = 1 / a * K_mm
            K_agent_k = K_nm[idx_sel[k]]
            H += 1 / sigma**2 * K_agent_k.T @ K_agent_k
            if beta_penalize is not None:
                H += beta_penalize * np.eye(m)
            # b = 1 / σ**2 * K_agent_k.T @ y_agent_k - sum_{p ~ k} A_kp.T @ λ_kp
            y_agent_k = y_n[idx_sel[k]]
            b = 1 / sigma**2 * K_agent_k.T @ y_agent_k
            # now add the λ mixing terms
            for p in range(a):
                if W[k, p] != 0 and k < p:
                    b -= lambda_agents[k, p]
                elif W[k, p] != 0 and k > p:
                    b += lambda_agents[k, p]
            # solve
            alpha_agents[k] = np.linalg.solve(H, b)
        # store alpha iteration
        alpha_seq.append(alpha_agents.copy())
        ### global lambda update
        for k in range(a):
            for p in range(a):
                if W[k, p] != 0:
                    lambda_agents[k, p] += step_size * (
                        alpha_agents[k] - alpha_agents[p]
                    )
        # store lambda iteration
        lambda_seq.append(lambda_agents.copy())

    return alpha_seq, lambda_seq


def run_ADMM(
    alpha_agents_0: np.ndarray,
    z_agents_0: np.ndarray,
    t_max: int,
    a: int,
    K_mm: np.ndarray,
    y_n: np.ndarray,
    idx_sel: np.ndarray,
    K_nm: np.ndarray,
    sigma: float,
    m: int,
    W: np.ndarray,
    step_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ADMM algorithm."""

    alpha_seq: list[np.ndarray] = [alpha_agents_0]
    z_seq: list[np.ndarray] = [z_agents_0]
    # alpha{,z}_agents will hold the current value of the alpha vectors
    alpha_agents = alpha_agents_0.copy()
    z_agents = z_agents_0.copy()

    for _ in trange(t_max):
        ### local alpha update (exact gradient solving)
        # -> solve H * α_k = b to find α_k
        # s.t. ∇L_k(α_k, lambda_k) = 0
        for k in range(a):
            # H = 1 / a * K_mm + 1 / σ**2 * K_agent_k.T @ K_agent_k + Σ_{p~k} βI_m
            H = 1 / a * K_mm
            K_agent_k = K_nm[idx_sel[k]]
            H += 1 / sigma**2 * K_agent_k.T @ K_agent_k
            for p in range(a):
                if W[k, p] != 0:
                    H += step_size * np.eye(m)
            # b = ...
            y_agent_k = y_n[idx_sel[k]]
            b = 1 / sigma**2 * K_agent_k.T @ y_agent_k
            for p in range(a):
                if W[k, p] != 0:
                    b += step_size * z_agents[k, p]
            # solve
            alpha_agents[k] = np.linalg.solve(H, b)
        # store alpha iteration
        alpha_seq.append(alpha_agents.copy())
        ### global z update
        for k in range(a):
            for p in range(a):
                z_agents[k, p] = (alpha_agents[k] + alpha_agents[p]) / 2
        # store z iteration
        z_seq.append(z_agents.copy())

    return alpha_seq, z_seq


def local_SGD(
    a: int,
    K_mm: np.ndarray,
    alpha_0: np.ndarray,
    E: int,
    x: np.ndarray,
    B: int,
    y: np.ndarray,
    K_agent: np.ndarray,
    sigma: float,
    step_size_seq: list[float],
) -> np.ndarray:
    """Performs local SGD on the agent's data."""
    # checks
    assert x.size == y.size
    assert len(x.shape) == len(y.shape) == 1
    # initialize alpha_0 for this agent with the server's current value
    alpha = alpha_0.copy()
    tot_data_size = x.size  # d
    for t in range(E):
        # common to all batches
        grad = (1 / a) * K_mm @ alpha
        for batch_lim in range(0, tot_data_size, B):
            # select the batch
            y_batch = y[batch_lim : batch_lim + B]  # B
            K_agent_batch = K_agent[batch_lim : batch_lim + B]  # B x m
            # compute gradient for this batch
            grad_add = K_agent_batch.T @ (y_batch - K_agent_batch @ alpha)
            grad -= grad_add / (sigma**2)
            # gradient descent step with variable step size
            alpha -= step_size_seq[t] * grad
    return alpha


def run_FedAvg(
    alpha_server_0: np.ndarray,
    t_max: int,
    rng: "np.random.Generator",
    a: int,
    c: int,
    m: int,
    K_dm_per_agent: np.ndarray,
    K_mm: np.ndarray,
    E: int,
    x: np.ndarray,
    B: int,
    y: np.ndarray,
    sigma: float,
    step_size_seq: list[float],
) -> list[np.ndarray]:
    """Federated Averaging algorithm."""
    alpha_seq = [alpha_server_0]
    # alpha_server will hold the current value of the alpha vector
    alpha_server = alpha_server_0.copy()
    # run the algorithm
    for _ in trange(t_max):
        # select c agents
        sel_agents = rng.choice(a, c, replace=False)
        alpha_agents = np.zeros((c, m))
        # run local SGD on each agent's data
        for idx_sel_agent, global_agent_idx in enumerate(sel_agents):
            K_agent = K_dm_per_agent[global_agent_idx]
            x_agent = x[global_agent_idx]
            y_agent = y[global_agent_idx]
            alpha_agents[idx_sel_agent] = local_SGD(
                a,
                K_mm,
                alpha_server,
                E,
                x_agent,
                B,
                y_agent,
                K_agent,
                sigma,
                step_size_seq,
            )
        # mix the alpha vectors
        alpha_server = np.mean(alpha_agents, axis=0)
        # store alpha iteration
        alpha_seq.append(alpha_server.copy())

    return alpha_seq
