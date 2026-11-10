"""
Dubins 4D avoid problem with a grid-based signed distance function (SDF) as l(s).

Same setup as examples/dubins_4d_avoid.py (grid, dynamics, horizon, solver options);
the only change is the initial value function: instead of CylinderShape, l(s) is the
SDF of an indoor layout (walls, tables, shelves). The SDF is computed the way it would
be from a real occupancy-grid map:

    occupancy grid (fine, its own resolution)
      -> Euclidean distance transform -> signed distance in meters
      -> bilinear resampling onto the odp (x, y) grid
      -> broadcast over the v and theta dimensions

Outputs are written to results/dubins_4d_avoid_sdf/ (see save_results below).
"""
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

import heterocl as hcl
from odp.Grid import Grid
from odp.dynamics import DubinsCar4D2
from odp.solver import HJSolver

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dubins_4d_avoid_sdf")

# ---------------------------------------------------------------------------
# Indoor layout (meters). The room fills the odp (x, y) domain [-3, 3] x [-1, 4].
# ---------------------------------------------------------------------------
WALL = 0.15  # wall thickness
OBSTACLES = [
    # outer walls
    dict(name="wall (west)",   kind="box", center=(-3.0 + WALL / 2, 1.5), size=(WALL, 5.0)),
    dict(name="wall (east)",   kind="box", center=(3.0 - WALL / 2, 1.5),  size=(WALL, 5.0)),
    dict(name="wall (south)",  kind="box", center=(0.0, -1.0 + WALL / 2), size=(6.0, WALL)),
    dict(name="wall (north)",  kind="box", center=(0.0, 4.0 - WALL / 2),  size=(6.0, WALL)),
    # interior partition with a 1.0 m doorway at y in [1.2, 2.2]
    dict(name="partition (lower)", kind="box", center=(1.075, 0.1), size=(WALL, 2.2)),
    dict(name="partition (upper)", kind="box", center=(1.075, 3.1), size=(WALL, 1.8)),
    # furniture
    dict(name="round table",  kind="circle", center=(-1.6, 2.4), radius=0.5),
    dict(name="square table", kind="box", center=(-0.9, 0.3), size=(0.8, 0.8)),
    dict(name="shelf (north wall)", kind="box", center=(-0.1, 3.65), size=(1.2, 0.4)),
    dict(name="shelf (east room)",  kind="box", center=(2.65, 0.6),  size=(0.4, 1.6)),
    dict(name="shelf (west wall)",  kind="box", center=(-2.65, 0.4), size=(0.4, 1.0)),
]
ROBOT_RADIUS = 0.0      # set > 0 to inflate obstacles by the robot footprint: l = sdf - r

MAP_RES = 0.02          # occupancy grid cell size (m)
MAP_PAD = 0.5           # map extends this far beyond the domain; padding counts as occupied


def sdf_box(X, Y, center, size):
    """Exact signed distance to an axis-aligned box."""
    qx = np.abs(X - center[0]) - size[0] / 2
    qy = np.abs(Y - center[1]) - size[1] / 2
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside


def sdf_circle(X, Y, center, radius):
    return np.hypot(X - center[0], Y - center[1]) - radius


def analytic_union_sdf(X, Y):
    """Union of the shapes (min of their SDFs); exact outside the obstacles."""
    d = np.full(X.shape, np.inf)
    for ob in OBSTACLES:
        if ob["kind"] == "box":
            d = np.minimum(d, sdf_box(X, Y, ob["center"], ob["size"]))
        else:
            d = np.minimum(d, sdf_circle(X, Y, ob["center"], ob["radius"]))
    return d


def exact_sdf(X, Y, g):
    """Exact SDF of the layout with everything outside the room domain treated as solid."""
    dom_center = ((g.min[0] + g.max[0]) / 2, (g.min[1] + g.max[1]) / 2)
    dom_size = (g.max[0] - g.min[0], g.max[1] - g.min[1])
    return np.minimum(analytic_union_sdf(X, Y), -sdf_box(X, Y, dom_center, dom_size))


def sdf_from_occupancy(occ, res):
    """Signed distance (m) from a boolean occupancy grid: + in free space, - inside obstacles.

    distance_transform_edt measures center-to-center distances to the nearest cell of the
    other class; subtracting res/2 puts the zero level on the cell edges.
    """
    d_out = distance_transform_edt(~occ) * res
    d_in = distance_transform_edt(occ) * res
    return np.where(occ, -(d_in - res / 2), d_out - res / 2)


def build_l(g):
    """Occupancy map -> SDF -> l(s) on the odp grid. Returns (l, map_data)."""
    x0, y0 = g.min[0] - MAP_PAD, g.min[1] - MAP_PAD
    x1, y1 = g.max[0] + MAP_PAD, g.max[1] + MAP_PAD
    map_x = np.arange(x0 + MAP_RES / 2, x1, MAP_RES)   # cell centers
    map_y = np.arange(y0 + MAP_RES / 2, y1, MAP_RES)
    MX, MY = np.meshgrid(map_x, map_y, indexing="ij")

    # Rasterize the layout into an occupancy grid (stand-in for a real map);
    # everything outside the room domain counts as occupied.
    sdf_true = exact_sdf(MX, MY, g)
    occ = sdf_true <= 0

    sdf_map = sdf_from_occupancy(occ, MAP_RES)

    # Bilinear resampling onto the odp (x, y) grid points, then lift to 4D
    interp = RegularGridInterpolator((map_x, map_y), sdf_map)
    GX, GY = np.meshgrid(g.grid_points[0], g.grid_points[1], indexing="ij")
    l_xy = interp(np.stack([GX, GY], axis=-1)) - ROBOT_RADIUS
    l = np.broadcast_to(l_xy[:, :, None, None], tuple(g.pts_each_dim)).copy()

    map_data = dict(map_x=map_x, map_y=map_y, occupancy=occ, sdf_map=sdf_map,
                    sdf_analytic=sdf_true, l_xy=l_xy)
    return l, map_data


class Tee:
    """Mirror stdout into a buffer so solver statistics can be parsed afterwards."""
    def __init__(self, stream):
        self.stream, self.lines = stream, []

    def write(self, s):
        self.stream.write(s)
        self.lines.append(s)

    def flush(self):
        self.stream.flush()


if __name__ == "__main__":
    # ---------------- Same setup as examples/dubins_4d_avoid.py ----------------
    grid_min, grid_max = np.array([-3.0, -1.0, 0.0, -math.pi]), np.array([3.0, 4.0, 4.0, math.pi])
    g = Grid(grid_min, grid_max, 4, np.array([60, 60, 20, 36]), [3])
    my_car = DubinsCar4D2(uMode="max", dMode="min")

    lookback_length = 1.5
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
    compMethods = {"TargetSetMode": "minVWithV0"}

    # ---------------- The only change: l(s) from a grid-based SDF ----------------
    t0 = time.perf_counter()
    Initial_value_f, map_data = build_l(g)
    t_sdf = time.perf_counter() - t0
    print("SDF built in {:.3f} s; map {} cells at {} m".format(t_sdf, map_data["occupancy"].shape, MAP_RES))

    tee = Tee(sys.stdout)
    sys.stdout = tee
    t0 = time.perf_counter()
    try:
        result = HJSolver(my_car, g, Initial_value_f, tau, compMethods,
                          saveAllTimeSteps=True, accuracy="medium")
    finally:
        sys.stdout = tee.stream
    t_solver = time.perf_counter() - t0

    log = "".join(tee.lines)
    kernel_time = float(log.split("Total kernel time (s):")[1].split()[0])
    n_substeps = log.count("Computational time to integrate")

    # ---------------- Save results ----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, arr in map_data.items():
        np.save(os.path.join(OUT_DIR, key + ".npy"), arr)
    np.save(os.path.join(OUT_DIR, "V_terminal.npy"), result[..., 0].astype(np.float32))   # t = 1.5 s
    v_index = 10                                                                               # v = 2.105 m/s
    np.save(os.path.join(OUT_DIR, "V_vslice_all_t.npy"), result[:, :, v_index].astype(np.float32))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    with open("/proc/cpuinfo") as f:
        cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), platform.processor())

    info = dict(
        # max_input = what was passed to Grid(); max = after odp drops the periodic endpoint.
        # Rebuild the grid from max_input, or the periodic dimension gets shrunk twice.
        grid=dict(min=g.min.tolist(), max_input=grid_max.tolist(), max=g.max.tolist(),
                  pts=g.pts_each_dim.tolist(), periodic=g.pDim, dx=g.dx.tolist()),
        dynamics=dict(name="DubinsCar4D2", uMin=my_car.uMin, uMax=my_car.uMax,
                      dMin=my_car.dMin, dMax=my_car.dMax, uMode=my_car.uMode, dMode=my_car.dMode,
                      wheelbase=0.3),
        tau=dict(lookback_length=lookback_length, t_step=t_step, n=len(tau)),
        compMethod=compMethods, accuracy="medium", v_index_saved=v_index,
        map=dict(res=MAP_RES, pad=MAP_PAD, shape=list(map_data["occupancy"].shape),
                 robot_radius=ROBOT_RADIUS),
        obstacles=OBSTACLES,
        timing=dict(sdf_build_s=t_sdf, hjsolver_wall_s=t_solver, kernel_s=kernel_time,
                    setup_compile_s=t_solver - kernel_time, n_substeps=n_substeps),
        unsafe_points=dict(t0=int((result[..., -1] <= 0).sum()), tT=int((result[..., 0] <= 0).sum()),
                           total=int(np.prod(g.pts_each_dim))),
        env=dict(cpu=cpu, threads=os.cpu_count(), python=platform.python_version(),
                 numpy=np.__version__, scipy=scipy.__version__,
                 heterocl=getattr(hcl, "__version__", "0.3"), odp_commit=commit),
    )
    with open(os.path.join(OUT_DIR, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\nSaved results to", os.path.normpath(OUT_DIR))
    print("SDF build {:.2f} s | HJSolver wall {:.2f} s (kernel {:.2f} s, {} sub-steps)".format(
        t_sdf, t_solver, kernel_time, n_substeps))
    print("Unsafe grid points: t=0 -> {}, t={} -> {} (of {})".format(
        info["unsafe_points"]["t0"], lookback_length, info["unsafe_points"]["tT"], info["unsafe_points"]["total"]))
