# From https://github.com/google/trajax/blob/main/tests/optimizers_test.py

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

from primal_dual_ilqr.optimizers import primal_dual_ilqr

from absl.testing import absltest
from absl.testing import parameterized
from functools import partial
from jax import numpy as np
from jax import config, disable_jit, jit, grad, jvp
from trajax.integrators import euler
from trajax.optimizers import objective, scipy_minimize

import jax


@jit
def acrobot(x, u, t, params=(1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0)):
    """Classic Acrobot system.

    Note this implementation emulates the OpenAI gym implementation of
    Acrobot-v2, which itself is based on Stutton's Reinforcement Learning book.

    Args:
      x: state, (4, ) array
      u: control, (1, ) array
      t: scalar time. Disregarded because system is time-invariant.
      params: tuple of (LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1, LINK_COM_POS_1,
        LINK_COM_POS_2 LINK_MOI_1, LINK_MOI_2)

    Returns:
      xdot: state time derivative, (4, )
    """
    del t  # Unused

    m1, m2, l1, lc1, lc2, I1, I2 = params
    g = 9.8
    a = u[0]
    theta1 = x[0]
    theta2 = x[1]
    dtheta1 = x[2]
    dtheta2 = x[3]
    d1 = (
        m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2))
        + I1
        + I2
    )
    d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
        + phi2
    )
    ddtheta2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])


class Test(parameterized.TestCase):
    def setUp(self):
        super(Test, self).setUp()

    def testSimpleLqrProblem(self):
        def dynamics(x, u, t):
            A = 6 * t * np.ones([2, 2]) + np.arange(4).reshape([2, 2])
            B = 6 * t * np.ones([2, 1]) + np.arange(4, 6).reshape([2, 1])
            c = 6 * t * np.ones(2) + np.arange(6, 8)
            return A @ x + B @ u + c

        def cost(x, u, t):
            Q = (t + 1) * np.diag(np.arange(1, 3))
            R = (t + 1) * np.array([3.0]).reshape([1, 1])
            q = -(t + 1) * np.arange(91, 93)
            r = -(t + 1) * np.array([99.0])
            M = (t + 1) / 100.0 * np.arange(55, 57).reshape([2, 1])
            return (
                0.5 * x.T @ Q @ x
                + 0.5 * u.T @ R @ u
                + x.T @ M @ u
                + q.T @ x
                + r.T @ u
            )

        x0 = np.array([-11.0, -22.0])
        T = 2
        n = 2
        m = 1
        X = np.zeros([T + 1, n])
        U = np.zeros([T, m])
        V = np.zeros_like(X)
        X, U, V, num_iterations, g, c, no_errors = primal_dual_ilqr(
            cost, dynamics, x0, X, U, V
        )
        print(f"{num_iterations=}, {g=}, c_norm={np.linalg.norm(c)}")
        self.assertTrue(no_errors)

    def testAcrobotSolve(self):
        T = 50
        goal = np.array([np.pi, 0.0, 0.0, 0.0])
        dynamics = euler(acrobot, dt=0.1)

        def cost(x, u, t, params):
            delta = x - goal
            terminal_cost = 0.5 * params[0] * np.dot(delta, delta)
            stagewise_cost = 0.5 * params[1] * np.dot(
                delta, delta
            ) + 0.5 * params[2] * np.dot(u, u)
            return np.where(t == T, terminal_cost, stagewise_cost)

        x0 = np.zeros(4)
        X = np.zeros((T + 1, 4))
        U = np.zeros((T, 1))
        V = np.zeros((T + 1, 4))
        params = np.array([1000.0, 0.1, 0.01])
        X, U, V, num_iterations, g, c, no_errors = primal_dual_ilqr(
            partial(cost, params=params), dynamics, x0, X, U, V
        )
        print(f"{num_iterations=}, {g=}, c_norm={np.linalg.norm(c)}")
        optimal_obj = 51.0
        self.assertLess(g, optimal_obj)
        self.assertTrue(no_errors)


if __name__ == "__main__":
    # Use float64 instead of float32.
    jax.config.update("jax_enable_x64", True)

    np.set_printoptions(threshold=1000000)
    np.set_printoptions(linewidth=1000000)

    absltest.main()
