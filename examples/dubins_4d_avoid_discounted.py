"""
Dubins 4D avoid problem with a discounted HJ variational inequality.

Same system and obstacle as examples/dubins_4d_avoid.py, solved with

    0 = min{ l(x) - V(x,t),  D_t V + max_u min_d D_x V . f(x,u,d) - gamma V }

via HJSolver(..., discount=gamma), for several gamma over a long horizon so the value converges.
In time-to-go tau the solution is V(x,tau) = max_u min_{s<=tau} e^{-gamma s} l(x(s)), so with l < 0
inside the obstacle:
    * viability kernel (can avoid forever): V -> 0 from above (flat 0 once converged, gamma > 0)
    * unsafe states: V < 0
    * {V <= 0} at a given horizon does not depend on gamma (e^{-gamma s} > 0) in exact arithmetic.

Two cases are solved:
    open  - the original example: cylinder obstacle, grid [-3,3] x [-1,4] x [0,4] x [-pi,pi).
            The grid edges are outflow boundaries (the car can drive off the grid, or brake /
            accelerate past v = 0 / 4), and odp extrapolates V away from zero into the ghost cells
            there. With gamma > 0 that acts as a permanent source of positive values, so the safe
            values do NOT flatten to 0.
    arena - "margin at every state boundary": the admissible set is the room interior and the speed
            range [V_ALLOWED_MIN, V_ALLOWED_MAX]; the grid extends past it by a margin in every
            non-periodic dimension, and l < 0 in that margin:
                l = min( cylinder,  distance to the (x, y) grid edge - WALL,
                         v - V_ALLOWED_MIN,  V_ALLOWED_MAX - v )
            With the original DubinsCar4D2 dynamics (no saturation), every grid edge then lies in
            a constraint region with V < 0, so nothing positive is extrapolated in and the
            discounted value converges to the flat-0 kernel.
            Solved with accuracy="medium" (2nd order, recommended). The 1st-order scheme (case
            "arena_low") is too conservative even at gamma = 0 and the discount erodes its safe set
            (~16% at gamma = 0.3, T = 10 s, vs ~5% for 2nd order). Caveat of 2nd order: after ~8 s the
            largest safe values stop decaying (a small plateau above e^{-gamma tau} max l) and the
            zero level keeps creeping slowly, so there is no strict convergence on this grid.

Outputs go to results/dubins_4d_avoid_discounted/<case>/ (see the end of the script).
"""
import json
import math
import os
import sys
import time

import numpy as np
from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.dynamics import DubinsCar4D2
from odp.solver import HJSolver

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dubins_4d_avoid_discounted")

# Default: one quick verification run (margin "arena" case). For the earlier comparison runs use e.g.
#   GAMMAS = [0.0, 0.3, 0.6, 0.9], LOOKBACK = 15.0 and  python ... open arena arena_low
GAMMAS = [0.3]
LOOKBACK = 10.0
T_STEP = 0.25       # output resolution (integration sub-steps still follow the CFL condition)
V_SLICE = 2.105     # saved speed slice (m/s), nearest grid point is used
CASES = ["arena"]

# arena margins: walls of thickness WALL inside the (x, y) grid, and a speed grid extended past the
# allowed range. The allowed range includes slow reverse so that "stop and wait" (v = 0) is not on
# the constraint boundary (there l would be 0 and V could never become positive).
WALL = 0.2                          # m
V_ALLOWED_MIN, V_ALLOWED_MAX = -0.2, 4.0
V_GRID = (-0.4, 4.4, 24)            # same speed spacing as the original grid (~0.21 m/s)
ACCURACY = {"open": "medium", "arena": "medium", "arena_low": "low"}


def make_grid(v_lo, v_hi, n_v):
    grid_min = np.array([-3.0, -1.0, v_lo, -math.pi])
    grid_max = np.array([3.0, 4.0, v_hi, math.pi])
    return Grid(grid_min, grid_max, 4, np.array([60, 60, n_v, 36]), [3]), grid_max


def box_distance(g):
    """Distance (m) to the nearest (x, y) edge of the grid, positive inside."""
    d_x = np.minimum(g.vs[0] - g.min[0], g.max[0] - g.vs[0])
    d_y = np.minimum(g.vs[1] - g.min[1], g.max[1] - g.vs[1])
    return np.minimum(d_x, d_y) + 0 * g.vs[2] + 0 * g.vs[3]


def speed_margin(g):
    """Signed distance (m/s) to the allowed speed range, positive inside it."""
    return np.minimum(g.vs[2] - V_ALLOWED_MIN, V_ALLOWED_MAX - g.vs[2]) + 0 * g.vs[0] + 0 * g.vs[1] + 0 * g.vs[3]


class Tee:
    """Mirror stdout into a buffer so the kernel time can be parsed afterwards."""
    def __init__(self, stream):
        self.stream, self.lines = stream, []

    def write(self, s):
        self.stream.write(s)
        self.lines.append(s)

    def flush(self):
        self.stream.flush()


def diagnostics(result, tau):
    """Per saved time: unsafe count, max/median V on {V > 0}, and max |dV/dtau| between saved times."""
    n = len(tau)
    rows, prev = [], None
    for k in range(n):
        V = result[..., n - 1 - k]          # result[..., -1] is tau = 0
        pos = V > 0
        row = dict(tau=float(tau[k]), unsafe=int((~pos).sum()),
                   max_pos=float(V[pos].max()) if pos.any() else 0.0,
                   median_pos=float(np.median(V[pos])) if pos.any() else 0.0,
                   min_V=float(V.min()))
        if prev is not None:
            dV = np.abs(V - prev)
            row["max_rate"] = float(dV.max() / (tau[k] - tau[k - 1]))
        rows.append(row)
        prev = V
    return rows


if __name__ == "__main__":
    compMethods = {"TargetSetMode": "minVWithV0"}
    tau = np.arange(start=0, stop=LOOKBACK + 1e-5, step=T_STEP)
    my_car = DubinsCar4D2(uMode="max", dMode="min")

    for case in (sys.argv[1:] or CASES):
        if case == "open":
            g, grid_max = make_grid(0.0, 4.0, 20)                 # same grid as dubins_4d_avoid.py
            Initial_value_f = CylinderShape(g, [2, 3], np.array([0., 2., 0., 0.]), 0.8)
        elif case in ("arena", "arena_low"):
            g, grid_max = make_grid(*V_GRID)
            Initial_value_f = np.minimum.reduce([
                CylinderShape(g, [2, 3], np.array([0., 2., 0., 0.]), 0.8),
                box_distance(g) - WALL,
                speed_margin(g)])
        else:
            raise ValueError("unknown case " + case)
        v_index = int(np.argmin(np.abs(g.grid_points[2] - V_SLICE)))

        out_dir = os.path.join(OUT_DIR, "{}_g{}_T{:g}".format(case, "-".join(str(gm) for gm in GAMMAS), LOOKBACK))

        # check: every non-periodic grid edge must lie in a constraint (l < 0), else edges can pin V > 0
        edges = [Initial_value_f[0], Initial_value_f[-1], Initial_value_f[:, 0], Initial_value_f[:, -1],
                 Initial_value_f[:, :, 0], Initial_value_f[:, :, -1]]
        edge_max = max(float(e.max()) for e in edges)
        print("[check] max l on the x/y/v grid edges = {:.3f} -> {}".format(
            edge_max, "OK (state boundary is unsafe)" if edge_max < 0 else "NOT a margin: edges can hold V > 0"))
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "l.npy"), Initial_value_f[:, :, v_index].astype(np.float32))

        runs = []
        for gamma in GAMMAS:
            print("\n========== {}: gamma = {} ==========".format(case, gamma))
            tee = Tee(sys.stdout)
            sys.stdout = tee
            t0 = time.perf_counter()
            try:
                result = HJSolver(my_car, g, Initial_value_f, tau, compMethods,
                                  saveAllTimeSteps=True, accuracy=ACCURACY[case], discount=gamma)
            finally:
                sys.stdout = tee.stream
            wall = time.perf_counter() - t0
            log = "".join(tee.lines)

            tag = "g{:.2f}".format(gamma).replace(".", "p")
            np.save(os.path.join(out_dir, "V_final_{}.npy".format(tag)), result[..., 0].astype(np.float32))
            np.save(os.path.join(out_dir, "V_vslice_all_t_{}.npy".format(tag)),
                    result[:, :, v_index].astype(np.float32))
            runs.append(dict(
                gamma=gamma, tag=tag, hjsolver_wall_s=wall,
                kernel_s=float(log.split("Total kernel time (s):")[1].split()[0]),
                n_substeps=log.count("Computational time to integrate"),
                diagnostics=diagnostics(result, tau)))
            del result
            last = runs[-1]["diagnostics"][-1]
            print("{} gamma={}: wall {:.1f} s, unsafe at T: {}, max V on safe set: {:.2e}, median {:.2e}".format(
                case, gamma, wall, last["unsafe"], last["max_pos"], last["median_pos"]), flush=True)
            if gamma > 0:
                # exact solution: V <= e^{-gamma T} max l on the safe set, so it flattens toward 0
                bound = math.exp(-gamma * LOOKBACK) * float(Initial_value_f.max())
                within = last["max_pos"] <= 1.05 * bound
                print("[check] max V on safe set {:.2e} vs bound e^(-gamma T) max l = {:.2e} -> {}".format(
                    last["max_pos"], bound, "within bound (flattens)" if within else
                    "above bound: 2nd-order plateau (expected with accuracy='medium', see README)"), flush=True)

        margins = dict(wall=WALL, v_allowed=[V_ALLOWED_MIN, V_ALLOWED_MAX]) if case != "open" else None
        info = dict(
            case=case,
            grid=dict(min=g.min.tolist(), max_input=grid_max.tolist(), max=g.max.tolist(),
                      pts=g.pts_each_dim.tolist(), periodic=g.pDim, dx=g.dx.tolist()),
            obstacle=dict(kind="cylinder", center=[0.0, 2.0], radius=0.8, ignore_dims=[2, 3]),
            margins=margins, l_max=float(Initial_value_f.max()),
            dynamics=dict(name=type(my_car).__name__, uMin=my_car.uMin, uMax=my_car.uMax,
                          dMin=my_car.dMin, dMax=my_car.dMax, uMode=my_car.uMode, dMode=my_car.dMode),
            tau=dict(lookback_length=LOOKBACK, t_step=T_STEP, n=len(tau)),
            compMethod=compMethods, accuracy=ACCURACY[case], v_index_saved=v_index,
            total_points=int(np.prod(g.pts_each_dim)), runs=runs)
        with open(os.path.join(out_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=1)
        print("\nSaved {} results to {}".format(case, os.path.normpath(out_dir)), flush=True)
