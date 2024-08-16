from jax import jit, lax, vmap

import jax.numpy as np

from functools import partial

from trajax.optimizers import vectorize

from primal_dual_ilqr.optimizers import primal_dual_ilqr


@partial(
    jit,
    static_argnums=(
        0,
        1,
        6,
        7,
    ),
)
def constrained_primal_dual_ilqr(
    cost,
    dynamics,
    x0,
    X_in,
    U_in,
    V_in,
    equality_constraint=lambda x, u, t: np.empty(0),
    inequality_constraint=lambda x, u, t: np.empty(0),
    max_iterations=100,
    max_al_iterations=5,
    slope_threshold=1e-4,
    var_threshold=0.0,
    c_sq_threshold=1e-4,
    make_psd=True,
    psd_delta=1e-6,
    armijo_factor=1e-4,
    alpha_0=1.0,
    alpha_mult=0.5,
    alpha_min=5e-5,
    complementary_slackness_threshold=1.0e-2,
    penalty_init=1.0,
    penalty_update_rate=10.0,
):
    """Implements the Primal-Dual iLQR algorithm, using Augmented Lagrangian for handling constraints.

    Args:
      cost:                   cost function with signature cost(x, u, t).
      dynamics:               dynamics function with signature dynamics(x, u, t).
      x0:                     [n]           numpy array.
      X_in:                   [T+1, n]      numpy array.
      U_in:                   [T, m]        numpy array.
      V_in:                   [T+1, n]      numpy array.
      equality_constraint:    equality_constraint(x, u, t) == 0 returns
                              (num_equality, ) nd array.
      inequality_constraint:  inequality_constraint(x, u, t) <= 0 returns
                              (num_inequality, ) nd array.
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
      iterations_ilqr:   the number of iLQR iterations upon convergence.
      iterations_al:     the number of AL iterations upon convergence.
      no_errors:         whether no errors were encountered during the solve.
    """

    T = U_in.shape[0]
    t_range = np.arange(T + 1)

    # augmented Lagrangian methods
    def augmented_lagrangian(x, u, t, dual_equality, dual_inequality, penalty):
        # stage cost
        J = cost(x, u, t)

        # stage equality constraint
        equality = equality_constraint(x, u, t)

        # stage inequality constraint
        inequality = inequality_constraint(x, u, t)

        # active set
        active_set = np.invert(np.isclose(dual_inequality[t], 0.0) & (inequality < 0.0))

        # update cost
        J += dual_equality[t].T @ equality + 0.5 * penalty * equality.T @ equality
        J += dual_inequality[t].T @ inequality + 0.5 * penalty * inequality.T @ (
            active_set * inequality
        )

        return J

    def dual_update(constraint, dual, penalty):
        return dual + penalty * constraint

    def inequality_projection(dual):
        return np.maximum(dual, 0.0)

    # vectorize
    equality_constraint_mapped = vectorize(equality_constraint)
    inequality_constraint_mapped = vectorize(inequality_constraint)
    dual_update_mapped = vmap(dual_update, in_axes=(0, 0, None))

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    # evaluate constraints
    U_pad = pad(U_in)
    equality_constraints = equality_constraint_mapped(X_in, U_pad, t_range)
    inequality_constraints = inequality_constraint_mapped(X_in, U_pad, t_range)

    # initialize dual variables
    dual_equality = np.zeros_like(equality_constraints)
    dual_inequality = np.zeros_like(inequality_constraints)

    # initialize penalty
    penalty = penalty_init

    def body(inputs):
        # unpack
        (
            X,
            U,
            V,
            dual_equality,
            dual_inequality,
            penalty,
            equality_constraints,
            inequality_constraints,
            _,
            _,
            _,
            iteration_ilqr,
            iteration_al,
            _,
        ) = inputs

        # augmented Lagrangian parameters
        al_args = {
            "dual_equality": dual_equality,
            "dual_inequality": dual_inequality,
            "penalty": penalty,
        }

        # solve iLQR problem
        X, U, V, iteration, obj, c, no_errors = primal_dual_ilqr(
            partial(augmented_lagrangian, **al_args),
            dynamics,
            x0,
            X,
            U,
            V,
            max_iterations - iteration_ilqr,
            slope_threshold,
            var_threshold,
            c_sq_threshold,
            make_psd,
            psd_delta,
            armijo_factor,
            alpha_0,
            alpha_mult,
            alpha_min,
        )

        # evalute constraints
        U_pad = pad(U)

        equality_constraints = equality_constraint_mapped(X, U_pad, t_range)

        inequality_constraints = inequality_constraint_mapped(X, U_pad, t_range)
        inequality_constraints_projected = inequality_projection(inequality_constraints)

        equality_max_viol = (
            0
            if equality_constraints.size == 0
            else np.max(np.abs(equality_constraints))
        )
        inequality_max_viol = (
            0
            if inequality_constraints_projected.size == 0
            else np.max(inequality_constraints_projected)
        )

        max_constraint_violation = np.maximum(
            equality_max_viol,
            inequality_max_viol,
        )

        max_dynamics_violation_sq = np.sum(c * c)

        # augmented Lagrangian update
        dual_equality = dual_update_mapped(equality_constraints, dual_equality, penalty)

        dual_inequality = dual_update_mapped(
            inequality_constraints, dual_inequality, penalty
        )
        dual_inequality = inequality_projection(dual_inequality)

        penalty *= penalty_update_rate

        # increment
        iteration_ilqr += iteration
        iteration_al += 1

        return (
            X,
            U,
            V,
            dual_equality,
            dual_inequality,
            penalty,
            equality_constraints,
            inequality_constraints,
            max_constraint_violation,
            max_dynamics_violation_sq,
            obj,
            iteration_ilqr,
            iteration_al,
            no_errors,
        )

    def continuation_criteria(inputs):
        # unpack
        (
            _,
            _,
            _,
            _,
            dual_inequality,
            _,
            _,
            inequality_constraints,
            max_constraint_violation,
            max_dynamics_violation_sq,
            _,
            iteration_ilqr,
            iteration_al,
            no_errors,
        ) = inputs
        c_not_ok = np.logical_or(
            max_constraint_violation * max_constraint_violation > c_sq_threshold,
            max_dynamics_violation_sq > c_sq_threshold,
        )
        max_complementary_slack = np.max(
            np.abs(inequality_constraints * dual_inequality)
        )
        it_ok = np.logical_and(
            iteration_ilqr < max_iterations, iteration_al < max_al_iterations
        )
        # check maximum constraint violation and augmented Lagrangian iterations
        return np.logical_and(
            np.logical_and(it_ok, no_errors),
            np.logical_or(
                c_not_ok,
                max_complementary_slack > complementary_slackness_threshold,
            ),
        )

    (
        X,
        U,
        V,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        iteration_ilqr,
        iteration_al,
        no_errors,
    ) = lax.while_loop(
        continuation_criteria,
        body,
        (
            X_in,
            U_in,
            V_in,
            dual_equality,
            dual_inequality,
            penalty,
            equality_constraints,
            inequality_constraints,
            np.inf,
            np.inf,
            np.inf,
            0,
            0,
            True,
        ),
    )

    no_errors = np.logical_and(no_errors, iteration_al < max_al_iterations)

    return X, U, V, iteration_ilqr, iteration_al, no_errors
