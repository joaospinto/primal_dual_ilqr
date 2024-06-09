# From https://github.com/google/trajax/blob/main/benchmarks/ilqr_benchmark.py

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from trajax import optimizers

from primal_dual_ilqr.optimizers import primal_dual_ilqr


@jax.jit
def cartpole(state, action, timestep, params=(10.0, 1.0, 0.5)):
    """Classic cartpole system.

    Args:
      state: state, (4, ) array
      action: control, (1, ) array
      timestep: scalar time
      params: tuple of (MASS_CART, MASS_POLE, LENGTH_POLE)

    Returns:
      xdot: state time derivative, (4, )
    """
    del timestep  # Unused

    mc, mp, l = params
    g = 9.81

    q = state[0:2]
    qd = state[2:]
    s = jnp.sin(q[1])
    c = jnp.cos(q[1])

    H = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l * l]])
    C = jnp.array([[0.0, -mp * qd[1] * l * s], [0.0, 0.0]])

    G = jnp.array([[0.0], [mp * g * l * s]])
    B = jnp.array([[1.0], [0.0]])

    CqdG = jnp.dot(C, jnp.expand_dims(qd, 1)) + G
    f = jnp.concatenate(
        (qd, jnp.squeeze(-jsp.linalg.solve(H, CqdG, assume_a="pos")))
    )

    v = jnp.squeeze(jsp.linalg.solve(H, B, assume_a="pos"))
    g = jnp.concatenate((jnp.zeros(2), v))
    xdot = f + g * action

    return xdot


def angle_wrap(th):
    return (th) % (2 * jnp.pi)


def state_wrap(s):
    return jnp.array([s[0], angle_wrap(s[1]), s[2], s[3]])


def squish(u):
    return 5 * jnp.tanh(u)


horizon = 50
dt = 0.1
eq_point = jnp.array([0, jnp.pi, 0, 0])


def cost(x, u, t):
    # err = state_wrap(x - eq_point)
    err = x - eq_point
    cv = jnp.maximum(jnp.abs(u) - 5.0, 0.0)
    # stage_cost = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
    stage_cost = (
        0.1 * jnp.dot(err, err)
        + 0.01 * jnp.dot(u, u)
        + 1000.0 * jnp.dot(cv, cv)
    )
    final_cost = 1000 * jnp.dot(err, err)
    return jnp.where(t == horizon, final_cost, stage_cost)


def dynamics(x, u, t):
    # return x + dt * cartpole(x, squish(u), t)
    return x + dt * cartpole(x, u, t)

# Use float64 instead of float32.
jax.config.update("jax_enable_x64", True)

# jnp.set_printoptions(threshold=1000000)
# jnp.set_printoptions(linewidth=1000000)

x0 = jnp.array([0.0, 0.2, 0.0, -0.1])
U0 = jnp.zeros([horizon, 1])
X0 = optimizers.rollout(dynamics, U0, x0)
V0 = jnp.zeros([horizon + 1, 4])

X, U, obj, gradient, adjoints, lqr, it = optimizers.ilqr(
    cost, dynamics, x0, U0, maxiter=1000
)

print(f"Trajax iLQR result: {obj=} {it=}")

# with jax.disable_jit():
X, U, V, num_iterations, g, c, no_errors = primal_dual_ilqr(
    cost,
    dynamics,
    x0,
    X0,
    U0,
    V0,
    max_iterations=1000,
)

print(f"Primal-Dual iLQR result: {num_iterations=} {no_errors=} {g=}")
