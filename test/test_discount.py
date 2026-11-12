import numpy as np
import heterocl as hcl

from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.solver import HJSolver


class ShiftX2D:
    """x_dot = 1, y_dot = 0. With l = l(y), l is constant along every trajectory, so the
    discounted avoid value has a closed form. (Pure zero dynamics can't be used: odp's CFL
    step 0.8 / sum(alpha_i / dx_i) evaluates to -inf when every alpha is 0.)"""

    def opt_ctrl(self, t, state, spat_deriv):
        u1 = hcl.scalar(0, "u1")
        u2 = hcl.scalar(0, "u2")
        return (u1[0], u2[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(1.0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        return (x_dot[0], y_dot[0])


class DriftCar2D:
    """x_dot = 1, y_dot = u with |u| <= 0.5; the control maximizes the Hamiltonian (avoid)."""

    def opt_ctrl(self, t, state, spat_deriv):
        u = hcl.scalar(0.5, "u")
        in2 = hcl.scalar(0, "in2")
        with hcl.if_(spat_deriv[1] < 0):
            u[0] = -0.5
        return (u[0], in2[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(1.0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        y_dot[0] = uOpt[0]
        return (x_dot[0], y_dot[0])


def test_discount_matches_closed_form():
    """l constant along trajectories: V(x, tau) = e^{-gamma tau} l where l > 0, and V = l where l <= 0."""
    g = Grid(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 2, np.array([41, 41]))
    l = CylinderShape(g, [0], np.zeros(2), 1.0)      # l = |y| - 1, independent of x
    gamma, T = 0.5, 2.0
    tau = np.arange(0, T + 1e-5, 0.05)

    V = HJSolver(ShiftX2D(), g, l, tau, {"TargetSetMode": "minVWithV0"},
                 accuracy="medium", discount=gamma)

    exact = np.where(l > 0, np.exp(-gamma * T) * l, l)
    np.testing.assert_allclose(V, exact, atol=5e-4)


def test_discount_keeps_zero_sublevel_set():
    """e^{-gamma s} > 0, so the discount rescales V but must not change where V <= 0."""
    g = Grid(np.array([-3.0, -2.0]), np.array([2.0, 2.0]), 2, np.array([81, 65]))
    l = CylinderShape(g, [], np.zeros(2), 0.5)
    tau = np.arange(0, 1.5 + 1e-5, 0.05)
    comp = {"TargetSetMode": "minVWithV0"}

    V0 = HJSolver(DriftCar2D(), g, l, tau, comp, accuracy="medium", discount=0.0)
    Vg = HJSolver(DriftCar2D(), g, l, tau, comp, accuracy="medium", discount=0.5)

    unsafe0, unsafeg = V0 <= 0, Vg <= 0
    assert unsafe0.sum() > (l <= 0).sum()            # the tube really grew upstream of the disk
    assert (unsafe0 ^ unsafeg).sum() <= 0.01 * unsafe0.sum()
    # e^{-gamma s} pulls V toward 0 from both sides: smaller where V0 > 0, less negative where V0 < 0
    pos, neg = V0 > 1e-3, V0 < -1e-3
    assert np.all(Vg[pos] <= V0[pos] + 1e-4)
    assert np.all(Vg[neg] >= V0[neg] - 1e-4)
