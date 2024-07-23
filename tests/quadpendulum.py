# From https://github.com/google/trajax/blob/main/notebooks/l4dc/QuadPend.ipynb

# Copyright 2023 Google LLC
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

import jax.numpy as jnp
import jax

import numpy as np

from trajax import integrators
from trajax.experimental.sqp import util

from primal_dual_ilqr.optimizers import primal_dual_ilqr
from functools import partial

from jax import grad, jvp

n = 8
m = 2

Mass = 0.486
mass = 0.2 * Mass
grav = 9.81
l = 0.25
L = 2 * l
J = 0.00383
fric = 0.01

u_hover = 0.5 * (Mass + mass) * grav * jnp.ones((m,))

# State: q = (p_x, p_y, theta, phi, velocities)
# where theta: rotation angle of quad
# phi: rotation angle of pendulum, w.r.t. vertical (NOTE: not a relative angle)


def get_mass_matrix(q):
    phi = q[-1]
    M_q = jnp.array(
        [
            [Mass + mass, 0.0, 0.0, mass * L * jnp.cos(phi)],
            [0.0, Mass + mass, 0.0, mass * L * jnp.sin(phi)],
            [0.0, 0.0, J, 0.0],
            [
                mass * L * jnp.cos(phi),
                mass * L * jnp.sin(phi),
                0.0,
                mass * L * L,
            ],
        ]
    )
    return M_q


def get_mass_inv(q):
    phi = q[-1]
    a = Mass + mass
    b = mass * L * jnp.cos(phi)
    c = mass * L * jnp.sin(phi)
    d = mass * L * L
    den = (mass * L) ** 2.0 - a * d
    M_inv = jnp.array(
        [
            [(c * c - a * d) / (a * den), -(b * c) / (a * den), 0.0, (b / den)],
            [-(b * c) / (a * den), (b * b - a * d) / (a * den), 0.0, (c / den)],
            [0.0, 0.0, (1.0 / J), 0.0],
            [(b / den), (c / den), 0.0, -(a / den)],
        ]
    )
    return M_inv


kinetic = lambda q, q_dot: 0.5 * jnp.vdot(q_dot, get_mass_matrix(q) @ q_dot)
potential = lambda q: Mass * grav * q[1] + mass * grav * (q[1] - L * jnp.cos(q[-1]))
lag = lambda q, q_dot: kinetic(q, q_dot) - potential(q)
dL_dq = grad(lag, 0)


@jax.jit
def ode(x, u, t):
    del t
    q, q_dot = jnp.split(x, [4])
    # (M_q * q_ddot + M_dot * q_dot) - (dL_dq) = F_q
    M_q, M_dot = jvp(get_mass_matrix, (q,), (q_dot,))
    M_inv = get_mass_inv(q)
    torque_fric_pole = -fric * (q_dot[-1] - q_dot[-2])
    F_q = jnp.array(
        [
            -jnp.sum(u) * jnp.sin(q[2]),
            jnp.sum(u) * jnp.cos(q[2]),
            (u[0] - u[1]) * l - torque_fric_pole,
            torque_fric_pole,
        ]
    )
    q_ddot = M_inv @ (F_q + dL_dq(q, q_dot) - (M_dot @ q_dot))
    return jnp.concatenate((q_dot, q_ddot))


dt = 0.025
dynamics = integrators.euler(ode, dt)

from jax import random

key = random.PRNGKey(1234)

# Confirm mass matrix and inverse computation
q = random.uniform(key, shape=(4,))
np.allclose(get_mass_matrix(q) @ get_mass_inv(q), np.eye(4))

# Define Geometry

quad = (
    jnp.array([[-l, 0.0], [l, 0.0]]),
    jnp.array([[-l, 0.0], [-l, 0.3 * l]]),
    jnp.array([[l, 0.0], [l, 0.3 * l]]),
    jnp.array([[-1.3 * l, 0.3 * l], [-0.7 * l, 0.3 * l]]),
    jnp.array([[0.7 * l, 0.3 * l], [1.3 * l, 0.3 * l]]),
)

pos_0 = jnp.array([-2.5, 1.5, 0.0, 0])
# pos_0 = jnp.array([-3., 0.5, 0., 0])
pos_g = jnp.array([3.0, -1.5, 0.0, jnp.pi])

obs = [
    (jnp.array([-1.0, 0.5]), 0.5),
    (jnp.array([0.75, -1.0]), 0.75),
    (jnp.array([-2.0, -1.0]), 0.5),
    (jnp.array([2.0, 1.0]), 0.5),
]

world_range = (jnp.array([-4.0, -2.0]), jnp.array([4.0, 2.0]))


# Extract obstacle avoidance constraint
def get_closest_point(endp, p_o):
    """Get closest point between point and straight-line between endpoints."""
    x, y = endp
    t_ = jnp.vdot(p_o - x, y - x) / jnp.vdot(y - x, y - x)
    t_min = jnp.minimum(1.0, jnp.maximum(0.0, t_))
    p_min = x + t_min * (y - x)
    return p_min


def obs_constraint(q):
    pos = q[:2]
    theta = q[2]
    phi = q[-1]

    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    pos_c = pos + R @ jnp.array([0.0, 0.15 * l])
    pole = (pos, pos + jnp.array([L * jnp.sin(phi), -L * jnp.cos(phi)]))

    def avoid_obs(pos_c, pole, ob):
        delta_body = pos_c - ob[0]
        body_dist_sq = jnp.vdot(delta_body, delta_body) - (ob[1] + l) ** 2
        pole_p = get_closest_point(pole, ob[0])
        delta_pole = pole_p - ob[0]
        pole_dist_sq = jnp.vdot(delta_pole, delta_pole) - (ob[1] ** 2)
        return jnp.array([body_dist_sq, pole_dist_sq])

    return jnp.concatenate([avoid_obs(pos_c, pole, ob) for ob in obs])


# Constants
n, m, T = (8, 2, 160)

# Do angle wrapping on theta and phi
s1_ind = (2, 3)
state_wrap = util.get_s1_wrapper(s1_ind)

# Goal and terminal cost
goal = jnp.concatenate((pos_g, jnp.zeros((4,))))


@jax.jit
def cost(
    x,
    u,
    t,
    weights=(1.0, 1.0, 1.0),
    Q_T=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
):
    delta = state_wrap(x - goal)
    pos_cost = jnp.vdot(delta[:3], delta[:3]) + (1.0 + jnp.cos(x[3]))
    ctrl_cost = jnp.vdot(u - u_hover, u - u_hover)

    stage_cost = weights[0] * pos_cost + weights[1] * ctrl_cost
    term_cost = weights[2] * jnp.vdot(delta, jnp.array(Q_T) * delta)

    return jnp.where(t == T, 0.5 * term_cost, 0.5 * stage_cost)


@jax.jit
def state_constraint(x, t, theta_lim=jnp.pi / 2.0):
    # Require c_x(x[t], t) >= 0
    # theta \in [-theta_lim, theta_lim]
    theta_cons = jnp.array((x[2] + theta_lim, theta_lim - x[2]))

    # obs cons
    avoid_cons = obs_constraint(x[:4])

    # world_cons
    world_cons = jnp.concatenate((x[:2] - world_range[0], world_range[1] - x[:2]))

    return jnp.concatenate((theta_cons, world_cons, avoid_cons))


control_bounds = (
    0.1 * Mass * grav * jnp.ones((m,)),
    3.0 * Mass * grav * jnp.ones((m,)),
)


@jax.jit
def mod_cost(x, u, t, cost_params, cons_params):
    base_cost = cost(x, u, t, *cost_params)
    # return base_cost
    control_delta_lb = u + np.array((0.1 * Mass * grav, 3.0 * Mass * grav))
    control_delta_ub = np.array((0.1 * Mass * grav, 3.0 * Mass * grav)) - u
    base_cons = jnp.concatenate(
        # [state_constraint(x, t, *cons_params), control_delta_lb, control_delta_ub]
        [state_constraint(x, t, *cons_params)]
        # [control_delta_lb, control_delta_ub]
    )
    const_viol = jnp.minimum(base_cons, jnp.zeros_like(base_cons))
    weights, _ = cost_params
    return base_cost + 0.5 * weights[3] * np.sum(const_viol * const_viol)


# Solve
x0 = jnp.concatenate((pos_0, jnp.zeros((4,))))
U0 = jnp.tile(u_hover, (T, 1))

weights = (0.01, 0.05, 5.0, 1e2)
Q_T = (1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
theta_lim = 3.0 * jnp.pi / 4.0

cost_params = (weights, Q_T)
cons_params = (theta_lim,)
prob_params = (cost_params, cons_params)


@jax.jit
def inequality_constraint(x, u, t, cons_params):
    control_delta_lb = u + np.array((0.1 * Mass * grav, 3.0 * Mass * grav))
    control_delta_ub = np.array((0.1 * Mass * grav, 3.0 * Mass * grav)) - u
    return jnp.concatenate(
        [
            state_constraint(x, t, *cons_params),
            control_delta_lb,
            control_delta_ub,
        ]
    )


X0 = jnp.tile(x0, (T + 1, 1))
# X0 = x0.reshape([1, n]) + jnp.arange(T + 1).reshape([T + 1, 1]) * (goal - x0).reshape(
#     [1, n]
# )

from timeit import default_timer as timer


@jax.jit
def work():
    return primal_dual_ilqr(
        partial(mod_cost, cost_params=cost_params, cons_params=cons_params),
        dynamics,
        x0,
        X0,
        U0,
        np.zeros((T + 1, 8)),
        max_iterations=10000,
        psd_delta=1e-3,
    )


X, U, V, num_iterations, g, c, no_errors = work()
X.block_until_ready()

start = timer()

n = 100
for i in range(n):
    X, _, _, _, _, _, _ = work()
    X.block_until_ready()

end = timer()

t = (end - start) / n

print(f"{t=}, {num_iterations=}, {g=}, {no_errors=}")
