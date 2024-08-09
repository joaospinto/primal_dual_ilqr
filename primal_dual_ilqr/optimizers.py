from jax import debug, grad, jit, lax, scipy, vmap

import jax.numpy as np

from functools import partial

from trajax.optimizers import evaluate, linearize, quadratize

from .kkt_helpers import compute_search_direction_kkt, tvlqr_kkt

from .dual_tvlqr import dual_lqr, dual_lqr_backward, dual_lqr_gpu

from .linalg_helpers import (
    invert_symmetric_positive_definite_matrix,
    project_psd_cone,
)

from .primal_tvlqr import tvlqr, tvlqr_gpu, rollout, rollout_gpu


def lagrangian(cost, dynamics, x0):
    """Returns a function to evaluate the associated Lagrangian."""

    def fun(x, u, t, v, v_prev):
        c1 = cost(x, u, t)
        c2 = np.dot(v, dynamics(x, u, t))
        c3 = np.dot(v_prev, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3

    return fun


@jit
def regularize(Q, R, M, make_psd, psd_delta):
    """Regularizes the Q and R matrices.

    Args:
      Q:             [T+1, n, n]      numpy array.
      R:             [T, m, m]        numpy array.
      M:             [T+1, n, m]      numpy array.
      make_psd:      whether to zero negative eigenvalues after quadratization.
      psd_delta:     the minimum eigenvalue post PSD cone projection.

    Returns:
      Q:             [T+1, n, n]      numpy array.
      R:             [T, m, m]        numpy array.
    """
    T, n, m = M.shape
    psd = vmap(partial(project_psd_cone, delta=psd_delta))

    # This is done to ensure that the R are positive definite.
    R = lax.cond(make_psd, psd, lambda x: x, R)

    # This is done to ensure that the Q - M R^(-1) M^T are positive semi-definite.
    Rinv = vmap(lambda t: invert_symmetric_positive_definite_matrix(R[t]))(np.arange(T))
    MRinvMT = vmap(lambda t: M[t] @ Rinv[t] @ M[t].T)(np.arange(T))
    QMRinvMT = vmap(lambda t: Q[t] - MRinvMT[t])(np.arange(T))
    QMRinvMT = lax.cond(make_psd, psd, lambda x: x, QMRinvMT)
    Q_T = Q[T].reshape([1, n, n])
    Q_T = lax.cond(make_psd, psd, lambda x: x, Q_T)
    Q = np.concatenate([QMRinvMT + MRinvMT, Q_T])

    return Q, R


@partial(jit, static_argnums=(0, 1))
def compute_search_direction(
    cost,
    dynamics,
    x0,
    X,
    U,
    V,
    c,
    make_psd,
    psd_delta,
):
    """Computes the SQP search direction.

    Args:
      cost:          cost function with signature cost(x, u, t).
      dynamics:      dynamics function with signature dynamics(x, u, t).
      x0:            [n]           numpy array.
      X:             [T+1, n]      numpy array.
      U:             [T, m]        numpy array.
      V:             [T+1, n]      numpy array.
      c:             [T+1, n]      numpy array.
      make_psd:      whether to zero negative eigenvalues after quadratization.
      psd_delta:     the minimum eigenvalue post PSD cone projection.

    Returns:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      q: [T+1, n]  numpy array.
      r: [T, m]    numpy array.
    """
    T = U.shape[0]

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    quadratizer = quadratize(lagrangian(cost, dynamics, x0), argnums=5)
    Q, R_pad, M_pad = quadratizer(X, pad(U), np.arange(T + 1), pad(V[1:]), V)

    R = R_pad[:-1]
    M = M_pad[:-1]

    Q, R = regularize(Q, R, M, make_psd, psd_delta)

    linearizer = linearize(lagrangian(cost, dynamics, x0), argnums=5)
    q, r_pad = linearizer(X, pad(U), np.arange(T + 1), pad(V[1:]), V)
    r = r_pad[:-1]

    dynamics_linearizer = linearize(dynamics)
    A_pad, B_pad = dynamics_linearizer(X, pad(U), np.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    K, k, P, p = tvlqr(Q, q, R, r, M, A, B, c[1:])
    # K, k, P, p = tvlqr_gpu(Q, q, R, r, M, A, B, c[1:])
    dX, dU = rollout(K, k, c[0], A, B, c[1:])
    # dX, dU = rollout_gpu(K, k, c[0], A, B, c[1:])
    dV = dual_lqr(dX, P, p)
    # dV = dual_lqr_backward(Q, q, M, A, dX, dU)
    # dV = dual_lqr_gpu(Q, q, M, A, dX, dU)

    # new_dX, new_dU, new_dV, LHS, rhs = tvlqr_kkt(Q, q, R, r, M, A, B, c[1:], c[0])

    # candidate_sol = np.concatenate([dX.flatten(), dU.flatten(), dV.flatten()])
    # candidate_sol = np.concatenate([new_dX.flatten(), new_dU.flatten(), new_dV.flatten()])
    # error = LHS @ candidate_sol - rhs
    # debug.print(f"error_norm={np.linalg.norm(error)}")

    # return new_dX, new_dU, new_dV, q, r

    return dX, dU, dV, q, r


@jit
def merit_rho(c, dV):
    """Determines the merit function penalty parameter to be used.

    Args:
      c:             [T+1, n]  numpy array.
      dV:            [T+1, n]  numpy array.

    Returns:
        rho: the penalty parameter.
    """
    c2 = np.sum(c * c)
    dV2 = np.sum(dV * dV)
    return lax.select(c2 > 1e-12, 2.0 * np.sqrt(dV2 / c2), 1e-2)


@jit
def slope(dX, dU, dV, c, q, r, rho):
    """Determines the directional derivative of the merit function.

    Args:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      dV: [T+1, n] numpy array.
      c:  [T+1, n] numpy array.
      q:  [T+1, n] numpy array.
      r:  [T, m] numpy array.
      rho: the penalty parameter of the merit function.

    Returns:
        dir_derivative: the directional derivative.
    """
    # return np.sum(q * dX) + np.sum(r * dU) + np.sum(dV * c) - rho * np.sum(c * c)
    return np.sum(q * dX) + np.sum(r * dU) - rho * np.sum(c * c)


@partial(jit, static_argnums=(0, 1))
def line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,
    alpha_0,
    alpha_mult,
    alpha_min,
):
    """Performs a primal-dual line search on an augmented Lagrangian merit function.

    Args:
      merit_function:  merit function mapping V, g, c to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:       whether no error occurred during the line search.
    """

    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        debug.print(f"{new_merit=}, {current_merit=}, {alpha=}, {merit_slope=}")
        return np.logical_and(
            new_merit > current_merit + alpha * armijo_factor * merit_slope,
            alpha > alpha_min,
        )

    def body(inputs):
        _, _, _, _, _, _, alpha = inputs
        alpha *= alpha_mult
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in
        debug.print(f"X_new.norm={np.linalg.norm(X_new)}")
        debug.print(f"U_new.norm={np.linalg.norm(U_new)}")
        debug.print(f"V_new.norm={np.linalg.norm(V_new)}")
        debug.print(f"dX.norm={np.linalg.norm(dX)}")
        debug.print(f"dU.norm={np.linalg.norm(dU)}")
        debug.print(f"dV.norm={np.linalg.norm(dV)}")
        new_g, new_c = model_evaluator(X_new, U_new)
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = np.where(np.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    X, U, V, new_g, new_c, _, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, np.inf, alpha_0 / alpha_mult),
    )

    debug.print(
        f"{new_g=}, c_sq_norm={np.sum(new_c * new_c)}, {merit_slope=}, {alpha=}"
    )

    no_errors = alpha > alpha_min

    V = V_in + alpha * dV

    return X, U, V, new_g, new_c, no_errors


@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper(cost, dynamics, x0, X, U):
    """Evaluates the costs and constraints based on the provided primal variables.

    Args:
      cost:            cost function with signature cost(x, u, t).
      dynamics:        dynamics function with signature dynamics(x, u, t).
      x0:              [n]           numpy array.
      X:               [T+1, n]      numpy array.
      U:               [T, m]        numpy array.

    Returns:
      g: the cost value (a scalar).
      c: the constraint values (a [T+1, n] numpy array).
    """
    T = U.shape[0]

    costs = partial(evaluate, cost)
    g = np.sum(costs(X, np.pad(U, [[0, 1], [0, 0]])))

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    c = np.vstack([x0 - X[0], vmap(residual_fn)(np.arange(T))])

    return g, c


# @partial(jit, static_argnums=(0, 1))
def primal_dual_ilqr(
    cost,
    dynamics,
    x0,
    X_in,
    U_in,
    V_in,
    max_iterations=100,
    slope_threshold=1e-4,
    var_threshold=0.0,
    c_sq_threshold=1e-4,
    make_psd=True,
    psd_delta=1e-6,
    armijo_factor=1e-4,
    alpha_0=1.0,
    alpha_mult=0.5,
    alpha_min=5e-5,
):
    """Implements the Primal-Dual iLQR algorithm.

    Args:
      cost:            cost function with signature cost(x, u, t).
      dynamics:        dynamics function with signature dynamics(x, u, t).
      x0:              [n]           numpy array.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      max_iterations:  maximum iterations.
      slope_threshold: tolerance for stopping optimization.
      var_threshold:   tolerance on primal and dual variables for stopping optimization.
      c_sq_threshold:  tolerance on squared constraint violations for stopping optimization.
      make_psd:        whether to zero negative eigenvalues after quadratization.
      psd_delta:       the minimum eigenvalue post PSD cone projection.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]        numpy array, representing the optimal state trajectory.
      U: [T, m]          numpy array, representing the optimal control trajectory.
      V: [T+1, n]        numpy array, representing the optimal multiplier trajectory.
      num_iterations:    the number of iterations upon convergence.
      final_cost:        the cost at the optimal state and control trajectory.
      final_constraints: the constraints at the optimal state and control trajectory.
      no_errors:         whether no errors were encountered during the solve.
    """
    model_evaluator = partial(model_evaluator_helper, cost, dynamics, x0)

    @jit
    def merit_function(V, g, c, rho):
        return g + np.sum((V + 0.5 * rho * c) * c)

    @jit
    def direction_and_merit(X, U, V, g, c):
        # dX, dU, dV, q, r = compute_search_direction_kkt(
        #     cost,
        #     dynamics,
        #     x0,
        #     X,
        #     U,
        #     V,
        #     make_psd,
        #     psd_delta,
        # )

        dX, dU, dV, q, r = compute_search_direction(
            cost,
            dynamics,
            x0,
            X,
            U,
            V,
            c,
            make_psd,
            psd_delta,
        )

        rho = merit_rho(c, dV)

        merit = merit_function(V, g, c, rho)

        merit_slope = slope(
            dX,
            dU,
            dV,
            c,
            q,
            r,
            rho,
        )

        @jit
        def f(x):
            gg, cc = model_evaluator(X + x * dX, U + x * dU)
            return merit_function(V + x * dV, gg, cc, rho)

        auto_merit_slope = grad(f)(0.0)

        debug.print(f"{auto_merit_slope=}")
        debug.print(f"{merit_slope=}")
        debug.print(f"MERIT FUNCTION SLOPE ERROR: {auto_merit_slope - merit_slope}")

        # merit_slope = auto_merit_slope

        return dX, dU, dV, rho, merit, merit_slope

    def body(inputs):
        """Solves LQR subproblem and returns updated trajectory."""
        X, U, V, dX, dU, dV, iteration, _, g, c, rho, merit, merit_slope = inputs

        debug.print(f"Constraint violation norm: {np.sum(c * c)}")

        X_new, U_new, V_new, g_new, c_new, no_errors = line_search(
            partial(merit_function, rho=rho),
            model_evaluator,
            X,
            U,
            V,
            dX,
            dU,
            dV,
            merit,
            g,
            c,
            merit_slope,
            armijo_factor,
            alpha_0,
            alpha_mult,
            alpha_min,
        )

        debug.print(f"no_errors coming out of line search: {no_errors}")

        (
            dX_new,
            dU_new,
            dV_new,
            rho_new,
            merit_new,
            merit_slope_new,
        ) = direction_and_merit(X_new, U_new, V_new, g_new, c_new)

        return (
            X_new,
            U_new,
            V_new,
            dX_new,
            dU_new,
            dV_new,
            iteration + 1,
            no_errors,
            g_new,
            c_new,
            rho_new,
            merit_new,
            merit_slope_new,
        )

    def continuation_criterion(inputs):
        _, _, _, dX, dU, dV, iteration, no_errors, _, c, _, _, slope = inputs

        c_sq_norm = np.sum(c * c)
        slope_ok = np.abs(slope) > slope_threshold
        delta_norm_sq = np.sum(dX * dX) + np.sum(dU * dU) + np.sum(dV * dV)
        delta_norm_ok = delta_norm_sq > var_threshold**2
        c_ok = c_sq_norm > c_sq_threshold
        progress_ok = np.logical_or(np.logical_and(slope_ok, delta_norm_ok), c_ok)

        status_ok = np.logical_and(no_errors, iteration < max_iterations)

        return np.logical_and(status_ok, progress_ok)

    g, c = model_evaluator(X_in, U_in)

    dX, dU, dV, rho, merit, merit_slope = direction_and_merit(X_in, U_in, V_in, g, c)

    X, U, V, _, _, _, iteration, no_errors, g, c, _, _, merit_slope = lax.while_loop(
        continuation_criterion,
        body,
        (
            X_in,
            U_in,
            V_in,
            dX,
            dU,
            dV,
            0,
            True,
            g,
            c,
            rho,
            merit,
            merit_slope,
        ),
    )

    no_errors = np.logical_and(no_errors, iteration < max_iterations)

    return X, U, V, iteration, g, c, no_errors
