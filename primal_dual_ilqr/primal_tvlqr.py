import jax.numpy as np

from jax import jit, lax, scipy, vmap

from .linalg_helpers import solve_symmetric_positive_definite_system


def lqr_step(P, p, Q, q, R, r, M, A, B, c):
    """Single LQR Step.

    Args:
      P: [n, n] numpy array.
      p: [n]    numpy array.
      Q: [n, n] numpy array.
      q: [n]    numpy array.
      R: [m, m] numpy array.
      r: [m]    numpy array.
      M: [n, m] numpy array.
      A: [n, n] numpy array.
      B: [n, m] numpy array.
      c: [n]    numpy array.

    Returns:
      K, k: state feedback gain and affine term.
      P, p: updated matrices encoding quadratic value function.
    """
    symmetrize = lambda x: 0.5 * (x + x.T)

    AtP = A.T @ P
    AtPA = symmetrize(AtP @ A)
    BtP = B.T @ P
    BtPA = BtP @ A

    H = BtPA + M.T
    h = B.T @ p + BtP @ c + r

    G = symmetrize(R + BtP @ B)

    K_k = solve_symmetric_positive_definite_system(
        G, -np.hstack((H, h.reshape([-1, 1])))
    )

    K = K_k[:, :-1]
    k = K_k[:, -1]

    P = symmetrize(Q + AtPA + K.T @ H)
    p = q + A.T @ p + AtP @ c + K.T @ h

    return K, k, P, p


@jit
def tvlqr(Q, q, R, r, M, A, B, c):
    """Discrete-time Finite Horizon Time-varying LQR.

    Args:
      Q: [T+1, n, n]  numpy array.
      q: [T+1, n]     numpy array.
      R: [T, m, m]    numpy array.
      r: [T, m]       numpy array.
      M: [T, n, m]    numpy array.
      A: [T, n, n]    numpy array.
      B: [T, n, m]    numpy array.
      c: [T, n]       numpy array.

    Returns:
      K: [T, m, n]    Gains
      k: [T, m]       Affine terms (u_t = K[t] x_t + k[t])
      P: [T+1, n, n]  numpy array encoding initial value function.
      p: [T+1, n]     numpy array encoding initial value function.
    """

    T = Q.shape[0] - 1
    n = Q.shape[1]

    def f(carry, elem):
        P, p = carry
        t = elem

        K, k, P, p = lqr_step(P, p, Q[t], q[t], R[t], r[t], M[t], A[t], B[t], c[t])

        new_carry = (P, p)
        new_output = (K, k, P, p)

        return new_carry, new_output

    K, k, P, p = lax.scan(f, (Q[T], q[T]), np.arange(T), T, reverse=True)[1]

    return (
        K,
        k,
        np.concatenate([P, Q[T].reshape([1, n, n])]),
        np.concatenate([p, q[T].reshape([1, n])]),
    )


@jit
def tvlqr_gpu(Q, q, R, r, M, A, B, c):
    """Discrete-time Finite Horizon Time-varying LQR.

    This is a O(log T) parallel time complexity implementation, based on
    https://ieeexplore.ieee.org/document/9697418.

    Args:
      Q: [T+1, n, n] numpy array.
      q: [T+1, n]    numpy array.
      R: [T, m, m]   numpy array.
      r: [T, m]      numpy array.
      M: [T, n, m]   numpy array.
      A: [T, n, n]   numpy array.
      B: [T, n, m]   numpy array.
      c: [T, n]      numpy array.
      delta: Enforces positive definiteness by ensuring smallest eigenval > delta.

    Returns:
      K: [T, m, n] Gains
      k: [T, m] Affine terms (u_t = K[t] x_t + k[t])
      P: [T+1, n, n] numpy array encoding initial value function.
      p: [T+1, n] numpy array encoding initial value function.
    """
    T = Q.shape[0] - 1
    n = Q.shape[1]

    def fn(next, prev):
        def decompose(elem):
            return (
                elem[:n],
                elem[n],
                elem[n + 1 : 2 * n + 1],
                elem[2 * n + 1],
                elem[-n:],
            )

        A_l, c_l, C_l, p_l, P_l = decompose(prev)
        A_r, c_r, C_r, p_r, P_r = decompose(next)

        ArIClPr_inv = A_r @ np.linalg.inv(np.eye(n) + C_l @ P_r)
        AlTIPrCl_inv = A_l.T @ np.linalg.inv(np.eye(n) + P_r @ C_l)

        A_new = ArIClPr_inv @ A_l
        c_new = ArIClPr_inv @ (c_l - C_l @ p_r) + c_r
        C_new = ArIClPr_inv @ C_l @ A_r.T + C_r
        p_new = AlTIPrCl_inv @ (p_r + P_r @ c_l) + p_l
        P_new = AlTIPrCl_inv @ P_r @ A_l + P_l

        return np.concatenate(
            [
                A_new,
                c_new.reshape(1, n),
                C_new,
                p_new.reshape(1, n),
                P_new,
            ]
        )

    def chol_inv(t):
        f = scipy.linalg.cho_factor(R[t])
        m = R[t].shape[0]
        return scipy.linalg.cho_solve(f, np.eye(m))

    Rinv = vmap(chol_inv)(np.arange(T))
    BRinv = vmap(lambda t: B[t] @ Rinv[t])(np.arange(T))
    MRinv = vmap(lambda t: M[t] @ Rinv[t])(np.arange(T))

    elems = np.concatenate(
        [
            # The A matrices.
            np.concatenate(
                [
                    A - vmap(lambda t: BRinv[t] @ M[t].T)(np.arange(T)),
                    np.zeros([1, n, n]),
                ]
            ),
            # The c vectors (b, in the notation of https://ieeexplore.ieee.org/document/9697418).
            np.concatenate(
                [
                    (c - vmap(lambda t: BRinv[t] @ r[t])(np.arange(T))).reshape(
                        [T, 1, n]
                    ),
                    np.zeros([1, 1, n]),
                ]
            ),
            # The C matrices.
            np.concatenate(
                [
                    vmap(lambda t: BRinv[t] @ B[t].T)(np.arange(T)),
                    np.zeros([1, n, n]),
                ]
            ),
            # The p vectors (-eta, in the notation of https://ieeexplore.ieee.org/document/9697418).
            q.reshape([T + 1, 1, n])
            - np.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ r[t])(np.arange(T)).reshape([T, 1, n]),
                    np.zeros([1, 1, n]),
                ]
            ),
            # The P matrices (J, in the notation of https://ieeexplore.ieee.org/document/9697418).
            Q
            - np.concatenate(
                [
                    vmap(lambda t: MRinv[t] @ M[t].T)(np.arange(T)),
                    np.zeros([1, n, n]),
                ]
            ),
        ],
        axis=1,
    )

    result = lax.associative_scan(lambda r, l: vmap(fn)(r, l), elems, reverse=True)

    P = result[:, -n:, :]
    p = result[:, 2 * n + 1, :]

    def getKs(t):
        symmetrize = lambda x: 0.5 * (x + x.T)

        BtP = B[t].T @ P[t + 1]
        BtPA = BtP @ A[t]

        H = BtPA + M[t].T
        h = B[t].T @ p[t + 1] + BtP @ c[t] + r[t]

        G = symmetrize(R[t] + BtP @ B[t])

        f = scipy.linalg.cho_factor(G)
        K_k = scipy.linalg.cho_solve(f, -np.hstack((H, h.reshape([-1, 1]))))
        K = K_k[:, :-1]
        k = K_k[:, -1]

        return K, k

    K, k = vmap(getKs)(np.arange(T))

    return K, k, P, p


@jit
def rollout(K, k, x0, A, B, c):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""

    T, n = c.shape

    def f(carry, elem):
        t = elem

        x = carry
        u = K[t] @ x + k[t]
        next_x = A[t] @ x + B[t] @ u + c[t]

        new_carry = next_x
        new_output = (next_x, u)

        return new_carry, new_output

    (X, U) = lax.scan(f, x0, np.arange(T), T)[1]

    return (np.concatenate([x0.reshape([1, n]), X]), U)


@jit
def rollout_gpu(K, k, x0, A, B, c):
    """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""
    T, _, n = K.shape

    def fn(prev, next):
        F = prev[:-1]
        f = prev[-1]
        G = next[:-1]
        g = next[-1]
        return np.concatenate([G @ F, (g + G @ f).reshape([1, n])])

    get_elem = lambda t: np.concatenate(
        [A[t] + B[t] @ K[t], (c[t] + B[t] @ k[t]).reshape([1, n])]
    )
    elems = vmap(get_elem)(np.arange(T))
    comp = lax.associative_scan(lambda l, r: vmap(fn)(l, r), elems)
    X = np.concatenate(
        [
            x0.reshape(1, n),
            vmap(lambda t: comp[t, :-1, :] @ x0 + comp[t, -1, :])(np.arange(T)),
        ]
    )

    U = vmap(lambda t: K[t] @ X[t] + k[t])(np.arange(T))

    return X, U
