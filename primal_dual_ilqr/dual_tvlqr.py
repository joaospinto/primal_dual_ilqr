from jax import jit, lax, vmap

import jax.numpy as np


@jit
def dual_lqr(X, P, p):
    """Dual LQR solve.

    Args:
      P: [T+1, n, n]   numpy array.
      p: [T+1, n]      numpy array.
      X: [T+1, n]      numpy array.

    Returns:
      V: [T+1, n] numpy array.
    """
    Tp1 = X.shape[0]
    return vmap(lambda t: P[t] @ X[t] + p[t])(np.arange(Tp1))


@jit
def dual_lqr_backward(Q, q, M, A, X, U):
    """Dual LQR solve.

    Args:
      Q: [T+1, n, n]   numpy array.
      M: [T,   n, m]   numpy array.
      q: [T+1, n]      numpy array.
      A: [T,   n, n]   numpy array.
      X: [T+1, n]      numpy array.
      U: [T,   m]      numpy array.

    Returns:
      V: [T+1, n] numpy array.
    """
    n = X.shape[1]
    T = U.shape[0]

    V_T = Q[T] @ X[T] + q[T]

    elems = vmap(lambda t: (A[t].T, Q[t] @ X[t] + M[t] @ U[t] + q[t]))(
        np.arange(T)
    )

    def f(v, e):
        return e[0] @ v + e[1], v

    out = lax.scan(f, V_T, elems, T, reverse=True)

    return np.concatenate([out[0].reshape(1, n), out[1]])


@jit
def dual_lqr_gpu(Q, q, M, A, X, U):
    """Dual LQR solve.

    Args:
      Q: [T+1, n, n]   numpy array.
      M: [T,   n, m]   numpy array.
      q: [T+1, n]      numpy array.
      A: [T,   n, n]   numpy array.
      X: [T+1, n]      numpy array.
      U: [T,   m]      numpy array.

    Returns:
      V: [T+1, n] numpy array.
    """
    n = X.shape[1]
    T = U.shape[0]
    get_elem = lambda t: np.concatenate(
        [A[t].T, (Q[t] @ X[t] + M[T] @ U[t] + q[t]).reshape(1, n)]
    )
    elems = np.concatenate(
        [
            vmap(get_elem)(np.arange(T)),
            np.concatenate(
                [np.eye(n), (Q[T] @ X[T] + q[T]).reshape(1, n)]
            ).reshape(1, n + 1, n),
        ]
    )

    def fn(next, prev):
        A = prev[:-1]
        b = prev[-1]
        C = next[:-1]
        d = next[-1]
        return np.concatenate([A @ C, (b + A @ d).reshape(1, n)])

    out = lax.associative_scan(lambda r, l: vmap(fn)(r, l), elems, reverse=True)
    return out[:, -1, :]
