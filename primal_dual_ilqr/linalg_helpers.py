from functools import partial

from jax import numpy as np
from jax import jit, lax, scipy, vmap

from trajax.optimizers import project_psd_cone as project_psd_cone_lapack


@jit
def _get_psd_eigenvalue_shift_ub(Q):
    """Returns a value k that guarantees that Q + kI is PSD.
    Relies on https://en.wikipedia.org/wiki/Gershgorin_circle_theorem.
    """
    D = np.diag(Q)

    # The radii of the circles.
    Rs = np.sum(np.abs(Q), axis=1) - np.abs(D)

    # The centers of the circles.
    Cs = D

    # The minimum shifts of each circle. Note: C - R + S > 0 and S >= 0 => S > R - C and S >= 0
    Ss = (Rs - Cs).clip(min=0.0)

    return np.max(Ss)


@jit
def _get_acceptable_psd_eigenvalue_shift(Q, k, delta):
    n, _ = Q.shape

    already_psd = is_positive_definite(Q, delta=delta)

    def continuation_criterion(k):
        return np.logical_and(
            np.logical_not(already_psd),
            is_positive_definite(Q + k * np.eye(n), delta=delta),
        )

    def body(k):
        return 0.5 * k

    k = 2.0 * lax.while_loop(
        continuation_criterion,
        body,
        k,
    )

    return lax.select(already_psd, 0.0, k)


@partial(jit, static_argnames=("use_lapack", "iterate"))
def project_psd_cone(Q, delta=0.0, use_lapack=False, iterate=True):
    """Projects to the cone of positive semi-definite matrices.

    Args:
      Q: [n, n] symmetric matrix.
      delta: minimum eigenvalue of the projection.

    Returns:
      [n, n] symmetric matrix projection of the input.
    """
    if use_lapack:
        return project_psd_cone_lapack(Q, delta=delta)

    n = Q.shape[0]

    k = _get_psd_eigenvalue_shift_ub(Q)

    if iterate:
        k = _get_acceptable_psd_eigenvalue_shift(Q, k, delta)

    return Q + k * np.eye(n)


@jit
def ldlt(Q):
    """Computes the L D L^T decomposition of Q."""
    n, _ = Q.shape
    if n == 1:
        return np.ones_like(Q), Q.reshape([1])
    else:
        L_prev, D_diag_prev = ldlt(Q[: n - 1, : n - 1])

        i = n - 1

        def f(carry, elem):
            partial_new_L = carry
            j = elem

            terms = vmap(lambda k: partial_new_L[k] * L_prev[j, k] * D_diag_prev[k])(
                np.arange(n - 1)
            )

            new_L_elem = (1.0 / D_diag_prev[j]) * (Q[i, j] - np.sum(terms))

            new_output = new_L_elem
            new_carry = vmap(
                lambda k: np.where(np.equal(k, j), new_L_elem, partial_new_L[k])
            )(np.arange(n - 1))

            return new_carry, new_output

        new_L_row = lax.scan(f, np.zeros(n - 1), np.arange(n - 1), n - 1)[1]

        L = np.block([[L_prev, np.zeros([n - 1, 1])], [new_L_row, np.array([1.0])]])

        terms = vmap(lambda j: L[i, j] * L[i, j] * D_diag_prev[j])(np.arange(i))

        new_D_elem = Q[i, i] - np.sum(terms)

        D_diag = np.append(D_diag_prev, new_D_elem)

        return L, D_diag


@jit
def is_positive_definite(Q, delta=0.0):
    """Checks whether the matrix Q is positive-definite.
    Does a L D L^T decomposition and checks that the diagonal entries of D are positive.
    See these for reference:
    1. https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition
    2. https://services.math.duke.edu/~jdr/2021f-218/materials/week11.pdf

    Args:
      Q: [n, n] symmetric matrix.
      delta: minimum eigenvalue of the projection.

    Returns:
      [n, n] symmetric matrix projection of the input.
    """
    L, D_diag = ldlt(Q)
    return np.all(D_diag > delta)


@jit
def _2x2_inv(M):
    # See https://en.wikipedia.org/wiki/Adjugate_matrix.
    a, b, c, d = M.flatten()
    det = a * d - b * c
    return (1.0 / det) * np.array([[d, -b], [-c, a]])


@jit
def solve_lower_unitriangular(L, b):
    """Solves Lx=b for x, where L is lower uni-triangular."""

    n, _ = L.shape

    def f(carry, elem):
        partial_new_x = carry
        i = elem

        new_x_elem = b[i] - np.dot(L[i, :], partial_new_x)

        new_output = new_x_elem
        new_carry = vmap(
            lambda k: np.where(np.equal(k, i), new_x_elem, partial_new_x[k])
        )(np.arange(n))

        return new_carry, new_output

    return lax.scan(f, np.zeros_like(b), np.arange(n), n)[1]


@jit
def solve_upper_unitriangular(U, b):
    """Solves Ux=b for x, where U is upper uni-triangular."""

    n, _ = U.shape

    def f(carry, elem):
        partial_new_x = carry
        i = elem

        new_x_elem = b[i] - np.dot(U[i, :], partial_new_x)

        new_output = new_x_elem
        new_carry = vmap(
            lambda k: np.where(np.equal(k, i), new_x_elem, partial_new_x[k])
        )(np.arange(n))

        return new_carry, new_output

    return lax.scan(f, np.zeros_like(b), np.arange(n), n, reverse=True)[1]


@partial(jit, static_argnames=("use_lapack",))
def solve_cholesky(A, b, use_lapack=False):
    n, _ = A.shape
    if use_lapack:
        f = scipy.linalg.cho_factor(A)
        return scipy.linalg.cho_solve(f, b)
    L, D_diag = ldlt(A)
    z = solve_lower_unitriangular(L, b)
    y = np.diag(1.0 / D_diag) @ z
    return solve_upper_unitriangular(L.T, y)


@partial(jit, static_argnames=("use_lapack",))
def solve_symmetric_positive_definite_system(A, b, use_lapack=False):
    n, _ = A.shape
    if n == 2:
        return _2x2_inv(A) @ b
    return solve_cholesky(A, b, use_lapack)


@partial(jit, static_argnames=("use_lapack",))
def invert_symmetric_positive_definite_matrix(M, use_lapack=False):
    n, _ = M.shape
    return solve_symmetric_positive_definite_system(M, np.eye(n), use_lapack)
