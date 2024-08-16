from jax import grad, hessian, jit, lax, vmap

import jax.numpy as np

from functools import partial

from trajax.optimizers import project_psd_cone


@partial(jit, static_argnums=(0, 1))
def compute_search_direction_kkt(
    cost,
    dynamics,
    x0,
    X_in,
    U_in,
    V_in,
    make_psd,
    psd_delta,
):
    """Computes the SQP search direction.

    Args:
      cost:                   cost function with signature cost(x, u, t).
      dynamics:               dynamics function with signature dynamics(x, u, t).
      x0:                     [n]           numpy array.
      X_in:                   [T+1, n]      numpy array.
      U_in:                   [T, m]        numpy array.
      V_in:                   [T+1, n]      numpy array.
      make_psd:               whether to zero negative eigenvalues after quadratization.
      psd_delta:              the minimum eigenvalue post PSD cone projection.

    Returns:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      dV: [T+1, n] numpy array.
      q: [T+1, n]   numpy array.
      r: [T, m]   numpy array.
    """
    T, m = U_in.shape

    def full_lagrangian(z):
        X = z[: X_in.size].reshape(X_in.shape)
        U = z[X_in.size : -V_in.size].reshape(U_in.shape)
        V = z[-V_in.size :].reshape(V_in.shape)
        f = lambda t: cost(X[t], U[t], t)
        cost_term = np.sum(vmap(f)(np.arange(T))) + cost(X[T], np.zeros(m), T)
        g = lambda t: np.dot(V[t + 1], dynamics(X[t], U[t], t) - X[t + 1])
        dynamics_term = np.sum(vmap(g)(np.arange(T)))
        initial_state_term = np.dot(V[0], x0 - X[0])
        return cost_term + dynamics_term + initial_state_term

    p = np.concatenate([X_in.flatten(), U_in.flatten(), V_in.flatten()])
    rhs = -grad(full_lagrangian)(p)
    LHS = hessian(full_lagrangian)(p)
    psd = partial(project_psd_cone, delta=psd_delta)
    N = X_in.size + U_in.size
    Q = lax.select(make_psd, psd(LHS[:N, :N]), LHS[:N, :N])
    LHS = LHS.at[:N, :N].set(Q)
    sol = np.linalg.solve(LHS, rhs)

    dX = sol[: X_in.size].reshape(X_in.shape)
    dU = sol[X_in.size : -V_in.size].reshape(U_in.shape)
    dV = sol[-V_in.size :].reshape(V_in.shape)

    q = -rhs[: dX.size].reshape(dX.shape)
    r = -rhs[dX.size : -dV.size].reshape(dU.shape)

    return dX, dU, dV, q, r


@partial(jit, static_argnums=(0, 1, 2))
def compute_search_direction_barrier_kkt(
    cost,
    dynamics,
    inequality_constraint,
    x0,
    X_in,
    U_in,
    V_in,
    Z_in,
    S_in,
    mu,
    make_psd,
    psd_delta,
):
    """Computes the SQP search direction.

    Args:
      cost:                   cost function with signature cost(x, u, t).
      dynamics:               dynamics function with signature dynamics(x, u, t).
      inequality_constraint:  inequality_constraint(x, u, t) <= 0 returns
                              (num_inequality, ) nd array.
      x0:                     [n]           numpy array.
      X_in:                   [T+1, n]      numpy array.
      U_in:                   [T, m]        numpy array.
      V_in:                   [T+1, n]      numpy array.
      Z_in:                   [T+1, c_dim]      numpy array.
      S_in:                   [T+1, c_dim]      numpy array.
      mu:                     a scalar parameter weighing the barrier term in the objective.
      make_psd:               whether to zero negative eigenvalues after quadratization.
      psd_delta:              the minimum eigenvalue post PSD cone projection.

    Returns:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      dV: [T+1, n] numpy array.
      dZ: [T+1, c_dim] numpy array.
      dS: [T+1, c_dim] numpy array.
    """
    T, m = U_in.shape

    def full_lagrangian(z):
        start_idx, end_idx = 0, X_in.size
        X = z[start_idx: end_idx].reshape(X_in.shape)
        start_idx, end_idx = end_idx, end_idx + U_in.size
        U = z[start_idx : end_idx].reshape(U_in.shape)
        start_idx, end_idx = end_idx, end_idx + V_in.size
        V = z[start_idx : end_idx].reshape(V_in.shape)
        start_idx, end_idx = end_idx, end_idx + S_in.size
        S = z[start_idx : end_idx].reshape(S_in.shape)
        start_idx, end_idx = end_idx, end_idx + Z_in.size
        Z = z[start_idx : end_idx].reshape(Z_in.shape)
        f = lambda t: cost(X[t], U[t], t)
        cost_term = np.sum(vmap(f)(np.arange(T))) + cost(X[T], np.zeros(m), T)
        g = lambda t: np.dot(V[t + 1], dynamics(X[t], U[t], t) - X[t + 1])
        dynamics_term = np.sum(vmap(g)(np.arange(T)))
        initial_state_term = np.dot(V[0], x0 - X[0])
        barrier_term = -mu * np.sum(vmap(lambda t: np.log(S[t]))(np.arange(T + 1)))
        h = lambda t: np.dot(Z[t], inequality_constraint(X[t], U[t], t) + S[t])
        inequality_term = np.sum(vmap(h)(np.arange(T + 1)))
        return cost_term + dynamics_term + initial_state_term + barrier_term + inequality_term

    p = np.concatenate([X_in.flatten(), U_in.flatten(), V_in.flatten(), Z_in.flatten(), S_in.flatten()])
    rhs = -grad(full_lagrangian)(p)
    LHS = hessian(full_lagrangian)(p)
    psd = partial(project_psd_cone, delta=psd_delta)
    N = X_in.size + U_in.size
    Q = lax.select(make_psd, psd(LHS[:N, :N]), LHS[:N, :N])
    LHS = LHS.at[:N, :N].set(Q)
    sol = np.linalg.solve(LHS, rhs)

    start_idx, end_idx = 0, X_in.size
    dX = sol[start_idx: end_idx].reshape(X_in.shape)
    start_idx, end_idx = end_idx, end_idx + U_in.size
    dU = sol[start_idx : end_idx].reshape(U_in.shape)
    start_idx, end_idx = end_idx, end_idx + V_in.size
    dV = sol[start_idx : end_idx].reshape(V_in.shape)
    start_idx, end_idx = end_idx, end_idx + S_in.size
    dS = sol[start_idx : end_idx].reshape(S_in.shape)
    start_idx, end_idx = end_idx, end_idx + Z_in.size
    dZ = sol[start_idx : end_idx].reshape(Z_in.shape)

    return dX, dU, dV, dZ, dS


@jit
def tvlqr_kkt(Q, q, R, r, M, A, B, c, x0):
    n = Q.shape[1]
    m = R.shape[1]
    T = R.shape[0]
    X_size = (T + 1) * n
    U_size = T * m
    V_size = X_size

    def lqr_lagrangian(z):
        X = z[:X_size].reshape([T + 1, n])
        U = z[X_size:-V_size].reshape([T, m])
        V = z[-V_size:].reshape([T + 1, n])
        f = (
            lambda t: np.dot(X[t], 0.5 * np.matmul(Q[t], X[t]) + q[t])
            + np.dot(U[t], 0.5 * np.matmul(R[t], U[t]) + r[t])
            + np.dot(X[t], np.matmul(M[t], U[t]))
        )
        cost = np.sum(vmap(f)(np.arange(T))) + np.dot(
            X[T], 0.5 * np.matmul(Q[T], X[T]) + q[T]
        )
        g = lambda t: np.dot(
            V[t + 1],
            np.matmul(A[t], X[t]) + np.matmul(B[t], U[t]) + c[t] - X[t + 1],
        )
        dynamics_term = np.sum(vmap(g)(np.arange(T)))
        initial_state_term = np.dot(V[0], x0 - X[0])
        return cost + dynamics_term + initial_state_term

    p = np.zeros(X_size + U_size + V_size)
    rhs = -grad(lqr_lagrangian)(p)
    LHS = hessian(lqr_lagrangian)(p)
    sol = np.linalg.solve(LHS, rhs)

    X = sol[: (T + 1) * n].reshape([T + 1, n])
    U = sol[X_size:-X_size].reshape([T, m])
    V = sol[-X_size:].reshape([T + 1, n])

    return X, U, V, LHS, rhs
