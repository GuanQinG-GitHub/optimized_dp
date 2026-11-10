"""
Build an HTML report for examples/dubins_4d_avoid_sdf.py from its saved results.

    python examples/dubins_4d_avoid_sdf.py          # solve + save results
    python examples/dubins_4d_avoid_sdf_report.py   # -> results/dubins_4d_avoid_sdf/report.html

The report is a single self-contained file (plotly.js and figures are inlined; only the
math typesetting, KaTeX, is loaded from a CDN).
"""
import base64
import io
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import plotly.graph_objects as go

from odp.Grid import Grid
from odp.Plots.plotting_utilities import BOX_GRID_AXIS, add_box_outline, slider_define

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dubins_4d_avoid_sdf")


def load(name):
    return np.load(os.path.join(RES_DIR, name + ".npy"))


info = json.load(open(os.path.join(RES_DIR, "run_info.json")))
map_x, map_y = load("map_x"), load("map_y")
occ, sdf_map, sdf_true, l_xy = load("occupancy"), load("sdf_map"), load("sdf_analytic"), load("l_xy")
V_T = load("V_terminal")            # (60, 60, 20, 36) at t = T
V_slice = load("V_vslice_all_t")    # (60, 60, 36, 31) at v index; [..., -1] is t = 0

gi = info["grid"]
g = Grid(np.array(gi["min"]), np.array(gi["max_input"]), 4, np.array(gi["pts"]), gi["periodic"])
assert np.allclose(g.max, gi["max"]), "grid rebuilt differently from the solver run"
xs, ys, vs, ths = g.grid_points
T = info["tau"]["lookback_length"]
dt_out = info["tau"]["t_step"]
v_idx = info["v_index_saved"]
dom = (gi["min"][0], gi["max"][0], gi["min"][1], gi["max"][1])
WALL_CM = round(100 * info["obstacles"][0]["size"][0])
map_extent = (map_x[0] - info["map"]["res"] / 2, map_x[-1] + info["map"]["res"] / 2,
              map_y[0] - info["map"]["res"] / 2, map_y[-1] + info["map"]["res"] / 2)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
OBST_CMAP = matplotlib.colors.ListedColormap([(1, 1, 1, 0), (0.42, 0.42, 0.42, 1)])


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_map(ax, alpha=1.0):
    ax.imshow(occ.T, origin="lower", extent=map_extent, cmap=OBST_CMAP, alpha=alpha,
              interpolation="nearest", zorder=1)
    ax.set_xlim(dom[0] - 0.05, dom[1] + 0.05)
    ax.set_ylim(dom[2] - 0.05, dom[3] + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


figs = {}

# --- Fig 1: layout / occupancy grid -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.4))
draw_map(ax)
# label position in free space (None = write inside the object)
LABEL_POS = {"round table": None, "square table": (-0.9, 1.2), "shelf (north wall)": (-0.1, 3.0),
             "shelf (west wall)": (-2.05, 1.4), "shelf (east room)": (1.85, -0.5),
             "partition (upper)": (2.0, 3.1), "partition (lower)": (0.3, 0.45)}
for ob in info["obstacles"]:
    if ob["name"] not in LABEL_POS:
        continue
    cx, cy = ob["center"]
    pos = LABEL_POS[ob["name"]]
    if pos is None:
        ax.annotate(ob["name"], (cx, cy), ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", zorder=3)
    else:
        ax.annotate(ob["name"], (cx, cy), xytext=pos, ha="center", va="center", fontsize=8, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#333"))
ax.annotate("wall ({} cm)".format(WALL_CM), (-0.5, -0.925), xytext=(-1.6, -0.55), ha="center", va="center",
            fontsize=8, zorder=3, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333"))
ax.annotate("doorway (1.0 m)", (1.075, 1.7), xytext=(1.6, 1.7), fontsize=8, va="center",
            arrowprops=dict(arrowstyle="->", lw=0.8), zorder=3)
ax.set_title("Occupancy grid: {} × {} cells at {:.0f} cm".format(occ.shape[0], occ.shape[1],
                                                                 100 * info["map"]["res"]))
figs["layout"] = png(fig)

# --- Fig 2: SDF on the map grid --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.4))
norm = TwoSlopeNorm(vmin=sdf_map.min(), vcenter=0.0, vmax=sdf_map.max())
im = ax.imshow(sdf_map.T, origin="lower", extent=map_extent, cmap="RdBu", norm=norm, interpolation="nearest")
MX, MY = np.meshgrid(map_x, map_y, indexing="ij")
cs = ax.contour(MX, MY, sdf_map, levels=np.arange(0.25, 2.0, 0.25), colors="k", linewidths=0.5, alpha=0.55)
ax.clabel(cs, fmt="%.2f", fontsize=6.5)
ax.contour(MX, MY, sdf_map, levels=[0.0], colors="k", linewidths=1.6)
ax.set_xlim(dom[0] - 0.05, dom[1] + 0.05)
ax.set_ylim(dom[2] - 0.05, dom[3] + 0.05)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
fig.colorbar(im, ax=ax, shrink=0.85, label="signed distance d(x, y) (m)")
ax.set_title("SDF from the occupancy grid (contours every 0.25 m, bold = 0)")
figs["sdf"] = png(fig)

# --- Fig 3: l(x, y) resampled on the odp grid ------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axs[0]
norm = TwoSlopeNorm(vmin=l_xy.min(), vcenter=0.0, vmax=l_xy.max())
dxh, dyh = g.dx[0] / 2, g.dx[1] / 2
im = ax.pcolormesh(np.r_[xs - dxh, xs[-1] + dxh], np.r_[ys - dyh, ys[-1] + dyh], l_xy.T,
                   cmap="RdBu", norm=norm, shading="flat", edgecolors=(1, 1, 1, 0.25), linewidth=0.2)
GX, GY = np.meshgrid(xs, ys, indexing="ij")
ax.contour(GX, GY, l_xy, levels=[0.0], colors="k", linewidths=1.4)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
fig.colorbar(im, ax=ax, shrink=0.85, label="l (m)")
ax.set_title("l(x, y) sampled at the {} × {} odp grid points".format(len(xs), len(ys)))
ax = axs[1]
draw_map(ax, alpha=0.35)
ax.contour(MX, MY, sdf_true, levels=[0.0], colors="k", linewidths=1.0, linestyles="--")
ax.contour(GX, GY, l_xy, levels=[0.0], colors="#c0392b", linewidths=1.4)
ax.plot([], [], "k--", lw=1.0, label="true obstacle boundary")
ax.plot([], [], color="#c0392b", lw=1.4, label="zero level of l on odp grid")
ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
ax.set_title("Boundary seen by the solver vs exact boundary")
fig.tight_layout()
figs["lgrid"] = png(fig)

# --- Fig 4: validation vs analytic SDF -------------------------------------------------
inside_dom = (MX >= dom[0]) & (MX <= dom[1]) & (MY >= dom[2]) & (MY <= dom[3])
free = inside_dom & (~occ)
err = np.abs(sdf_map - sdf_true)
err_free = err[free]
band = free & (np.abs(sdf_true) < 0.3)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
e = np.where(free, err, np.nan)
im = ax.imshow(e.T * 100, origin="lower", extent=map_extent, cmap="magma_r", interpolation="nearest",
               vmin=0, vmax=max(1e-6, np.nanpercentile(e, 99.9) * 100))
draw_map(ax, alpha=0.6)
fig.colorbar(im, ax=ax, shrink=0.85, label="|d_EDT − d_exact| (cm)")
ax.set_title("SDF error in free space")
ax = axs[1]
ax.hist(err_free * 100, bins=60, color="#2f6f5e")
ax.set_yscale("log")
ax.set_xlabel("|d_EDT − d_exact| (cm), free-space cells")
ax.set_ylabel("cell count")
ax.set_title("Error distribution")
fig.tight_layout()
figs["valid"] = png(fig)

# exact SDF at the odp (x, y) grid points, to quantify the resampling step too
import importlib.util
spec = importlib.util.spec_from_file_location("sdfex", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    "dubins_4d_avoid_sdf.py"))
sdfex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdfex)
l_exact = sdfex.exact_sdf(GX, GY, g) - info["map"]["robot_radius"]
dom_mask = l_exact > 0
stats = dict(
    err_free_max_cm=100 * err_free.max(),
    err_free_mean_cm=100 * err_free.mean(),
    err_band_max_cm=100 * err[band].max(),
    l_grid_max_cm=100 * np.abs(l_xy - l_exact)[dom_mask].max(),
    sign_mismatch=int(((l_xy <= 0) != (l_exact <= 0)).sum()),
)

# --- Fig 5: terminal-time slices at special headings ----------------------------------
def theta_index(deg):
    return int(np.argmin(np.abs(np.angle(np.exp(1j * (ths - math.radians(deg)))))))


headings = [0, 90, 180, -90]
fig, axs = plt.subplots(2, 2, figsize=(11, 9.4))
slice_rows = []
for ax, deg in zip(axs.ravel(), headings):
    k = theta_index(deg)
    Vs = V_T[:, :, v_idx, k]
    draw_map(ax)
    ax.contourf(GX, GY, Vs, levels=[-1e9, 0.0], colors=["#e74c3c"], alpha=0.28, zorder=2)
    ax.contour(GX, GY, Vs, levels=[0.0], colors="#c0392b", linewidths=1.6, zorder=3)
    ax.contour(GX, GY, l_xy, levels=[0.0], colors="k", linewidths=0.8, linestyles="--", zorder=3)
    th = ths[k]
    ax.annotate("", xy=(-2.45 + 0.45 * math.cos(th), 3.4 + 0.45 * math.sin(th)), xytext=(-2.45, 3.4),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#1f4e79"), zorder=5)
    free_g = l_xy > 0
    frac = ((Vs <= 0) & free_g).sum() / free_g.sum()
    slice_rows.append((deg, math.degrees(th), frac))
    ax.set_title("θ = {:+.0f}°  ·  v = {:.2f} m/s  ·  {:.0%} of free space unsafe".format(
        math.degrees(th), vs[v_idx], frac))
handles = [plt.Rectangle((0, 0), 1, 1, color=(0.42, 0.42, 0.42)),
           plt.Rectangle((0, 0), 1, 1, color="#e74c3c", alpha=0.28),
           plt.Line2D([], [], color="#c0392b", lw=1.6), plt.Line2D([], [], color="k", lw=0.8, ls="--"),
           plt.Line2D([], [], color="#1f4e79", lw=2)]
fig.legend(handles, ["obstacle (occupancy)", "unsafe: V(x, y, v, θ; T) ≤ 0", "zero level V = 0",
                     "obstacle as seen on odp grid (l = 0)", "heading θ"],
           loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
figs["slices"] = png(fig)

# --- Fig 6: speed dependence at theta = 0 ---------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.6))
draw_map(ax)
k0 = theta_index(0)
v_list = [0, 5, 10, 15, 19]
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(v_list)))
for c, vi in zip(colors, v_list):
    ax.contour(GX, GY, V_T[:, :, vi, k0], levels=[0.0], colors=[c], linewidths=1.5, zorder=3)
    ax.plot([], [], color=c, lw=1.5, label="v = {:.2f} m/s".format(vs[vi]))
ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)
ax.set_title("Zero level of V(·, ·, v, θ = 0°; T) for several speeds")
figs["speeds"] = png(fig)

# --- Fig 7: growth over time at the animated speed, theta = 0 -------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.6))
draw_map(ax)
t_list = [0.0, 0.5, 1.0, 1.5]
n_t = V_slice.shape[-1]
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(t_list)))
for c, t in zip(colors, t_list):
    idx = n_t - 1 - int(round(t / dt_out))
    ax.contour(GX, GY, V_slice[:, :, k0, idx], levels=[0.0], colors=[c], linewidths=1.5, zorder=3)
    ax.plot([], [], color=c, lw=1.5, label="t = {:.1f} s".format(t))
ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)
ax.set_title("Growth of the unsafe set over time (v = {:.2f} m/s, θ = 0°)".format(vs[v_idx]))
figs["growth"] = png(fig)

# --- 3D animation (same style as odp's plot_isosurface, lighter frames) -------------------
# Crop 2 grid cells off each (x, y) side so the outer walls (which sit in those cells) don't
# wall in the whole view; interior obstacles and near-wall unsafe bands stay visible.
CROP = 2
cx, cy = xs[CROP:-CROP], ys[CROP:-CROP]
g3 = Grid(np.array([cx[0], cy[0], -math.pi]), np.array([cx[-1], cy[-1], math.pi]),
          3, np.array([len(cx), len(cy), gi["pts"][3]]), [2])
X3, Y3, Z3 = np.meshgrid(g3.grid_points[0], g3.grid_points[1], g3.grid_points[2], indexing="ij")


def frame_values(idx):
    # Only the sign and values near 0 matter for the V = 0 isosurface: clip to +-0.5 m and store
    # integer centimeters, which keeps the zero crossing to within ~1 mm and shrinks the file
    V = V_slice[CROP:-CROP, CROP:-CROP, :, idx]
    return np.round(100 * np.clip(V, -0.5, 0.5)).astype(np.int16).ravel()


iso_style = dict(isomin=0, isomax=0, surface_count=1, colorscale="Rainbow", opacity=0.8,
                 caps=dict(x_show=False, y_show=False, z_show=False), showscale=False)
frames = [go.Frame(data=[go.Isosurface(value=frame_values(n_t - 1 - k))], traces=[0], name=str(k))
          for k in range(n_t)]
fig3 = go.Figure(frames=frames)
fig3.add_trace(go.Isosurface(x=np.round(X3.ravel(), 4), y=np.round(Y3.ravel(), 4), z=np.round(Z3.ravel(), 4),
                             value=frame_values(n_t - 1), **iso_style))
fig3.update_layout(
    title="Unsafe set V = 0 in (x, y, θ) at v = {:.2f} m/s".format(vs[v_idx]),
    height=680, margin=dict(l=10, r=10, t=50, b=10),
    scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="θ (rad)",
               camera_eye={"x": 1.35, "y": -1.55, "z": 0.95},
               aspectratio={"x": 1.2, "y": 1.0, "z": 0.7}))
fig3.update_scenes(xaxis=BOX_GRID_AXIS, yaxis=BOX_GRID_AXIS, zaxis=BOX_GRID_AXIS)
add_box_outline(fig3, g3)
fig3 = slider_define(fig3)
fig3.layout.sliders[0].currentvalue.prefix = "t = "
fig3.layout.sliders[0].currentvalue.suffix = " s"
for k, step in enumerate(fig3.layout.sliders[0].steps):
    step.label = "{:.2f}".format(k * dt_out)
anim_html = fig3.to_html(full_html=False, include_plotlyjs=True, div_id="anim3d",
                         config={"displaylogo": False})

# --- HTML -----------------------------------------------------------------------------------
tm = info["timing"]
up = info["unsafe_points"]
env = info["env"]
dyn = info["dynamics"]
obst_rows = "\n".join(
    "<tr><td>{}</td><td>{}</td><td>({:.3g}, {:.3g})</td><td>{}</td></tr>".format(
        ob["name"], ob["kind"], ob["center"][0], ob["center"][1],
        "{:.2f} × {:.2f} m".format(*ob["size"]) if ob["kind"] == "box" else "r = {:.2f} m".format(ob["radius"]))
    for ob in info["obstacles"])
slice_table = "\n".join("<tr><td>{:+.0f}°</td><td>{:.0%}</td></tr>".format(r[1], r[2]) for r in slice_rows)

CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --border:#e3e0d8; --code-bg:#f3f1ec;
        --accent:#2f6f5e; --accent-bg:#e6f2ee; color-scheme: light; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1060px; margin:0 auto; padding:32px 16px 80px; }
h1 { font-size:1.75rem; margin:0 0 4px; }
h2 { font-size:1.25rem; margin:44px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; scroll-margin-top:16px; }
h3 { font-size:1.02rem; margin:22px 0 8px; scroll-margin-top:16px; }
.meta { color:var(--muted); font-size:.9rem; }
code { font-family:ui-monospace,Consolas,monospace; background:var(--code-bg); padding:0 4px; border-radius:3px; font-size:.92em; }
pre { background:var(--code-bg); padding:10px 12px; border-radius:6px; overflow-x:auto; font:13px/1.5 ui-monospace,Consolas,monospace; }
figure { margin:16px 0 22px; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; }
figure img { width:100%; height:auto; display:block; }
figcaption { color:var(--muted); font-size:.88rem; margin-top:8px; }
figcaption b { color:var(--fg); }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; border-bottom:1px solid var(--border); padding:6px 8px; vertical-align:top; }
th { color:var(--muted); font-weight:600; }
.callout { background:var(--accent-bg); border-left:3px solid var(--accent); padding:10px 14px; border-radius:4px; margin:12px 0; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }
.kpi { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:12px 0; }
.kpi div { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px 12px; }
.kpi b { display:block; font-size:1.25rem; }
.kpi span { color:var(--muted); font-size:.8rem; }
.toc { position:fixed; top:16px; right:16px; width:240px; max-height:calc(100vh - 32px); overflow-y:auto; background:var(--card);
       border:1px solid var(--border); border-radius:10px; padding:12px 14px; box-shadow:0 6px 24px rgba(30,30,25,.10); font-size:.86rem; z-index:20; }
.toc strong { font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.toc ol { list-style:none; margin:6px 0 0; padding:0; }
.toc ol ol { padding-left:12px; border-left:1px solid var(--border); margin:2px 0 4px 6px; }
.toc a { display:block; padding:3px 8px; border-radius:5px; color:var(--fg); text-decoration:none; }
.toc ol ol a { color:var(--muted); padding:2px 8px; }
.toc a:hover { background:var(--code-bg); }
.toc a.active { background:var(--accent-bg); color:var(--accent); font-weight:600; }
@media (min-width:900px) { body > main { margin-right:280px; margin-left:max(0px, calc((100vw - 1060px - 280px) / 2)); } }
@media (max-width:899px) { .toc { position:static; width:auto; max-height:none; margin:16px 16px 0; box-shadow:none; } }
"""

TOC_JS = """
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  var targets = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
  function update() {
    var current = 0;
    for (var i = 0; i < targets.length; i++) { if (targets[i] && targets[i].getBoundingClientRect().top <= 80) current = i; }
    links.forEach(function (a, i) { a.classList.toggle('active', i === current); });
  }
  window.addEventListener('scroll', update, { passive: true });
  update();
})();
"""

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDF Avoid Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}]}});"></script>
<style>{CSS}</style>
</head>
<body>
<nav class="toc" aria-label="Contents"><strong>Contents</strong>
<ol>
<li><a href="#overview">Overview</a></li>
<li><a href="#sdf">1. The SDF obstacle map</a>
  <ol>
  <li><a href="#sdf-layout">Layout &amp; occupancy</a></li>
  <li><a href="#sdf-field">Signed distance field</a></li>
  <li><a href="#sdf-grid">l(s) on the odp grid</a></li>
  <li><a href="#sdf-valid">Validation</a></li>
  </ol></li>
<li><a href="#anim">2. 3D animation</a></li>
<li><a href="#slices">3. Terminal-time slices</a>
  <ol>
  <li><a href="#slices-heading">Special headings</a></li>
  <li><a href="#slices-speed">Speed dependence</a></li>
  <li><a href="#slices-time">Growth over time</a></li>
  </ol></li>
<li><a href="#math">4. Math &amp; numerics</a>
  <ol>
  <li><a href="#math-dyn">Dynamics</a></li>
  <li><a href="#math-sdf">SDF &amp; l(s)</a></li>
  <li><a href="#math-hj">HJ formulation</a></li>
  <li><a href="#math-num">Numerical scheme</a></li>
  <li><a href="#math-time">Computation time</a></li>
  </ol></li>
<li><a href="#repro">Reproduce</a></li>
</ol></nav>
<main>
<h1>SDF Avoid Report</h1>
<p class="meta"><code>examples/dubins_4d_avoid_sdf.py</code> · optimized_dp commit <code>{env['odp_commit']}</code> (with local, uncommitted changes) · generated by <code>examples/dubins_4d_avoid_sdf_report.py</code></p>

<h2 id="overview">Overview</h2>
<p>This is the same problem as <code>examples/dubins_4d_avoid.py</code>, with the same 4D car, grid, horizon and solver options. The only change is the initial value function \\(l(s)\\). Instead of one analytic cylinder, it is the <strong>signed distance function of an indoor layout</strong>, computed from an occupancy grid and sampled on the solver grid. The result is the backward reachable tube: the states from which the car <em>cannot avoid</em> touching an obstacle within \\(T = {T}\\) s, whatever it does with acceleration and steering.</p>
<div class="kpi">
<div><b>{tm['kernel_s']:.1f} s</b><span>solver kernel time</span></div>
<div><b>{tm['hjsolver_wall_s']:.1f} s</b><span>HJSolver wall time</span></div>
<div><b>{tm['sdf_build_s']*1000:.0f} ms</b><span>SDF construction</span></div>
<div><b>{up['t0']/up['total']:.0%} → {up['tT']/up['total']:.0%}</b><span>unsafe grid points, t = 0 → {T} s</span></div>
</div>

<h2 id="sdf">1. The SDF obstacle map</h2>
<h3 id="sdf-layout">Layout and occupancy grid</h3>
<p>The room fills the solver's \\((x, y)\\) domain \\([{dom[0]:g}, {dom[1]:g}] \\times [{dom[2]:g}, {dom[3]:g}]\\) m. It has {WALL_CM} cm walls, a partition with a 1 m doorway, two tables and three shelves. The layout is rasterized into a boolean occupancy grid at {info['map']['res']*100:.0f} cm resolution. This is the input a real mapping pipeline would provide. The map extends {info['map']['pad']} m past the domain, and that padding is marked occupied (solid exterior).</p>
<figure><img src="{figs['layout']}" alt="Occupancy grid of the indoor layout">
<figcaption><b>Figure 1.</b> Occupancy grid (grey = occupied). Walls are {WALL_CM} cm thick; furniture sizes are listed in the table below. The solver grid covers exactly this 6 m × 5 m area, with 60 × 60 points in \\((x, y)\\).</figcaption></figure>
<div class="wrap"><table>
<tr><th>Obstacle</th><th>Shape</th><th>Center (m)</th><th>Size</th></tr>
{obst_rows}
</table></div>

<h3 id="sdf-field">Signed distance field</h3>
<p>The SDF is computed only from the occupancy grid, with two Euclidean distance transforms (one for free cells, one for occupied cells). It is positive in free space, negative inside obstacles, and zero on obstacle boundaries, in meters.</p>
<figure><img src="{figs['sdf']}" alt="Signed distance field heat map">
<figcaption><b>Figure 2.</b> Signed distance \\(d(x, y)\\) on the {info['map']['res']*100:.0f} cm map grid (blue = free space, red = inside obstacles). Thin lines are iso-distance contours every 0.25 m, and the bold line is \\(d = 0\\). The value at a point is how far it is from the nearest obstacle surface, which is why the contours round off around convex corners.</figcaption></figure>

<h3 id="sdf-grid">l(s) on the solver grid</h3>
<p>The solver works on a coarser grid: \\(\\Delta x = {gi['dx'][0]:.3f}\\) m, \\(\\Delta y = {gi['dx'][1]:.3f}\\) m. The SDF is bilinearly interpolated at those 60 × 60 points, then copied along \\(v\\) and \\(\\theta\\). This assumes a point robot, i.e. obstacles not inflated (<code>ROBOT_RADIUS = {info['map']['robot_radius']}</code>).</p>
<figure><img src="{figs['lgrid']}" alt="l on the odp grid and boundary comparison">
<figcaption><b>Figure 3.</b> <b>Left:</b> \\(l(x, y)\\) as stored on the solver grid, one cell per grid point. <b>Right:</b> the zero level the solver effectively sees (red, linear interpolation between grid points) against the exact boundary (dashed). Corners round off at the scale of one grid cell (≈ 10 cm). A wall thinner than about two grid cells would start to blur, so the 15 cm walls here are near that limit.</figcaption></figure>

<h3 id="sdf-valid">Validation against the exact SDF</h3>
<p>Every shape here is a box or a circle, so the exact SDF is known in closed form. That lets us measure the error of the occupancy → EDT → SDF pipeline directly.</p>
<figure><img src="{figs['valid']}" alt="SDF error map and histogram">
<figcaption><b>Figure 4.</b> Absolute error between the EDT-based SDF and the exact SDF, in free space. In free space the max error is <b>{stats['err_free_max_cm']:.2f} cm</b> and the mean is {stats['err_free_mean_cm']:.2f} cm. Within 30 cm of an obstacle the max error is {stats['err_band_max_cm']:.2f} cm. Almost all of this comes from <em>rasterizing</em> the layout, not from the distance transform. Wall faces at e.g. \\(x = -2.85\\) fall mid-cell on the 2 cm map, so the occupied region is off by half a cell, which gives the flat 1.0 cm regions. Pale regions are where the boundary lands on a cell edge (≈ 0 error), and the radial streaks come from the round table's stair-stepped edge. A real occupancy map has the same half-cell limit. At the solver grid points, max \\(|l - l_{{exact}}| = {stats['l_grid_max_cm']:.2f}\\) cm in free space, and {stats['sign_mismatch']} grid points have a different inside/outside classification.</figcaption></figure>

<h2 id="anim">2. 3D animation: unsafe set over time</h2>
<p>This uses the same style as odp's <code>plot_isosurface</code>: a \\(V = 0\\) isosurface in \\((x, y, \\theta)\\) at a fixed speed \\(v = {vs[v_idx]:.2f}\\) m/s (<code>slicesCut=[{v_idx}]</code>), with Play/pause and a time slider. Drag to rotate, and scroll to zoom. At \\(t = 0\\) the surface is just the interior obstacles extruded along \\(\\theta\\) (tables, shelves, partition). As \\(t\\) grows, each obstacle's unsafe region swells toward the headings that point the car at it, which makes the twisted shapes along \\(\\theta\\). Bands also grow in from the walls. For visibility, the view is cropped two grid cells inside the outer walls (\\(x \\in [{cx[0]:.2f}, {cx[-1]:.2f}]\\), \\(y \\in [{cy[0]:.2f}, {cy[-1]:.2f}]\\)); otherwise the walls would enclose the whole scene. The surface caps are also off.</p>
<figure>{anim_html}
<figcaption><b>Figure 5.</b> Unsafe set \\(\\{{(x, y, \\theta) : V(x, y, {vs[v_idx]:.2f}, \\theta; t) \\le 0\\}}\\) for \\(t = 0, {dt_out}, \\dots, {T}\\) s. To keep this file small, frames store only the values: \\(V\\) clipped to ±0.5 m and stored as whole centimeters. Only the zero crossing is drawn, and it moves by at most about 1 mm.</figcaption></figure>

<h2 id="slices">3. Zero level set at the terminal time \\(T = {T}\\) s</h2>
<h3 id="slices-heading">Special headings</h3>
<figure><img src="{figs['slices']}" alt="Four 2D slices at headings 0, 90, 180, -90 degrees">
<figcaption><b>Figure 6.</b> 2D slices of \\(V(x, y, v, \\theta; T)\\) at \\(v = {vs[v_idx]:.2f}\\) m/s for headings 0°, 90°, 180° and −90° (the arrow shows the heading). Red = unsafe, meaning a collision can't be avoided within {T} s. Each unsafe region extends <em>upstream</em> of obstacles along the heading: a car pointing at a wall, too close to stop or turn away, is doomed. With max braking 1.5 m/s² the car needs about {vs[v_idx]**2/3:.2f} m to stop from {vs[v_idx]:.2f} m/s. The steering limit of ±10° makes turning radii large (\\(L/\\tan 10° \\approx 1.7\\) m).</figcaption></figure>
<div class="wrap" style="max-width:420px"><table>
<tr><th>Heading θ</th><th>Unsafe share of free space (v = {vs[v_idx]:.2f} m/s)</th></tr>
{slice_table}
</table></div>
<p class="meta">The θ grid has 10° spacing (36 points over [−π, π)), so 0°, ±90° and 180° lie exactly on grid points, while 45° does not.</p>

<h3 id="slices-speed">Speed dependence</h3>
<figure><img src="{figs['speeds']}" alt="Zero level at theta 0 for several speeds">
<figcaption><b>Figure 7.</b> Zero level at θ = 0° (facing +x) for several speeds. At \\(v = 0\\) the unsafe set is the obstacles themselves, plus a thin margin from the car's ability to accelerate toward them. It grows on the −x side of each obstacle as the speed rises.</figcaption></figure>

<h3 id="slices-time">Growth over time</h3>
<figure><img src="{figs['growth']}" alt="Zero level growth over time">
<figcaption><b>Figure 8.</b> Zero level at θ = 0°, \\(v = {vs[v_idx]:.2f}\\) m/s for \\(t = 0, 0.5, 1.0, 1.5\\) s. Because <code>minVWithV0</code> clamps \\(V \\le l\\), the sets are nested: this is a tube, not a set at a single time.</figcaption></figure>

<h2 id="math">4. Math behind the example</h2>
<h3 id="math-dyn">Dynamics (<code>DubinsCar4D2</code>)</h3>
<p>State \\(s = (x, y, v, \\theta)\\), control \\(u = (a, \\delta)\\) and disturbance \\(d = (d_1, d_2)\\):</p>
$$\\dot x = v\\cos\\theta + d_1,\\qquad \\dot y = v\\sin\\theta + d_2,\\qquad \\dot v = a,\\qquad \\dot\\theta = \\frac{{v\\,\\tan\\delta}}{{L}}$$
<p>with \\(L = {dyn['wheelbase']}\\) m, \\(a \\in [{dyn['uMin'][0]}, {dyn['uMax'][0]}]\\) m/s², \\(\\delta \\in [-\\pi/18, \\pi/18]\\) (±10°), and \\(d_1, d_2 \\in [{dyn['dMin'][0]}, {dyn['dMax'][0]}]\\). The disturbance is disabled, so this is an optimal control problem rather than a game.</p>

<h3 id="math-sdf">Signed distance function and \\(l(s)\\)</h3>
<p>For an obstacle region \\(\\mathcal O \\subset \\mathbb R^2\\), the signed distance function is</p>
$$d(p) = \\begin{{cases}} \\;\\;\\operatorname{{dist}}(p, \\mathcal O), & p \\notin \\mathcal O \\\\ -\\operatorname{{dist}}(p, \\mathbb R^2\\setminus\\mathcal O), & p \\in \\mathcal O \\end{{cases}}$$
<p>It is 1-Lipschitz, with \\(|\\nabla d| = 1\\) almost everywhere. From an occupancy grid \\(o_{{ij}} \\in \\{{0, 1\\}}\\) with cell size \\(h = {info['map']['res']}\\) m:</p>
$$\\hat d_{{ij}} = \\begin{{cases}} h\\,\\mathrm{{EDT}}(\\lnot o)_{{ij}} - h/2, & o_{{ij}} = 0 \\\\ -\\big(h\\,\\mathrm{{EDT}}(o)_{{ij}} - h/2\\big), & o_{{ij}} = 1 \\end{{cases}}$$
<p>Here \\(\\mathrm{{EDT}}(m)_{{ij}}\\) is the distance in cells from cell \\(ij\\) to the nearest cell where \\(m = 0\\) (<code>scipy.ndimage.distance_transform_edt</code>). The \\(h/2\\) shift puts the zero level on cell edges rather than cell centers. The target set, i.e. the obstacles in this avoid problem, is lifted to 4D by ignoring \\(v\\) and \\(\\theta\\):</p>
$$l(x, y, v, \\theta) = \\mathcal I[\\hat d](x, y) - r_{{robot}}, \\qquad \\mathcal T = \\{{s : l(s) \\le 0\\}}$$
<p>where \\(\\mathcal I\\) is bilinear interpolation from the map grid to the solver grid, and \\(r_{{robot}} = {info['map']['robot_radius']}\\).</p>

<h3 id="math-hj">Hamilton–Jacobi formulation</h3>
<p>The backward reachable tube over horizon \\(t\\) is the set of states from which every control eventually hits \\(\\mathcal T\\) within time \\(t\\):</p>
$$\\mathcal R(t) = \\{{ s_0 : \\forall u(\\cdot),\\ \\exists \\tau \\in [0, t],\\ s(\\tau) \\in \\mathcal T \\}} = \\{{ s : V(s, t) \\le 0 \\}}$$
<p>In time-to-go \\(t\\), odp solves</p>
$$\\frac{{\\partial V}}{{\\partial t}} = H(s, \\nabla V),\\qquad H(s, p) = \\max_{{u}}\\, \\min_{{d}}\\; p\\cdot f(s, u, d),\\qquad V(s, 0) = l(s)$$
<p>and after every step applies \\(V \\leftarrow \\min(V, l)\\) (<code>TargetSetMode = "minVWithV0"</code>), which gives a tube instead of a set at exactly time \\(t\\). The control maximizes \\(H\\) (<code>uMode = "max"</code>: stay away from \\(\\mathcal T\\)), and since \\(f\\) is affine in each control with box bounds, the optimum is bang-bang:</p>
$$a^* = \\begin{{cases}} a_{{max}}, & p_v \\ge 0 \\\\ a_{{min}}, & p_v < 0 \\end{{cases}}\\qquad \\delta^* = \\begin{{cases}} \\delta_{{max}}, & p_\\theta \\ge 0 \\\\ \\delta_{{min}}, & p_\\theta < 0 \\end{{cases}}\\quad (v \\ge 0 \\text{{ on this grid}})$$

<h3 id="math-num">Numerical scheme (as implemented in <code>graph_4D.py</code> / <code>solver.py</code>)</h3>
<div class="wrap"><table>
<tr><th>Item</th><th>Setting</th></tr>
<tr><td>Grid</td><td>\\(x \\in [{gi['min'][0]:g}, {gi['max'][0]:g}]\\), \\(y \\in [{gi['min'][1]:g}, {gi['max'][1]:g}]\\), \\(v \\in [{gi['min'][2]:g}, {gi['max'][2]:g}]\\), \\(\\theta \\in [-\\pi, \\pi)\\) periodic; {gi['pts'][0]} × {gi['pts'][1]} × {gi['pts'][2]} × {gi['pts'][3]} = {up['total']:,} points</td></tr>
<tr><td>Spacing</td><td>\\(\\Delta = ({gi['dx'][0]:.4f}, {gi['dx'][1]:.4f}, {gi['dx'][2]:.4f}, {gi['dx'][3]:.4f})\\) (θ step = 10°)</td></tr>
<tr><td>Spatial derivatives</td><td>2nd-order ENO, left/right one-sided \\(p^\\pm_i\\). Non-periodic edges use ghost values extrapolated away from zero.</td></tr>
<tr><td>Numerical Hamiltonian</td><td>Lax–Friedrichs: \\(\\hat H = H\\big(s, \\tfrac{{p^+ + p^-}}{{2}}\\big) + \\tfrac12 \\sum_i \\alpha_i \\,(p_i^+ - p_i^-)\\), with \\(\\alpha_i \\ge |f_i|\\) bounded over the optimal inputs at the min/max derivative values</td></tr>
<tr><td>Time integration</td><td>2nd-order TVD Runge–Kutta (Heun): \\(V^{{(1)}} = V^n + \\Delta t\\,\\hat H(V^n)\\), \\(V^{{(2)}} = V^{{(1)}} + \\Delta t\\,\\hat H(V^{{(1)}})\\), \\(V^{{n+1}} = \\tfrac12(V^n + V^{{(2)}})\\), then \\(V^{{n+1}} \\leftarrow \\min(V^{{n+1}}, l)\\)</td></tr>
<tr><td>Time step</td><td>CFL: \\(\\Delta t = \\min\\!\\big(0.8 \\,/\\, \\sum_i \\alpha_i^{{max}}/\\Delta_i,\\ t_{{k+1}} - t\\big)\\) → {tm['n_substeps']} sub-steps for {info['tau']['n']-1} output intervals of {dt_out} s</td></tr>
<tr><td>Precision / parallelism</td><td>float32 kernels compiled by HeteroCL 0.3 (LLVM); outer loop parallelized over threads</td></tr>
</table></div>

<h3 id="math-time">Computation time</h3>
<div class="wrap"><table>
<tr><th>Stage</th><th>Time</th><th>Notes</th></tr>
<tr><td>Occupancy → SDF → \\(l(s)\\)</td><td>{tm['sdf_build_s']*1000:.1f} ms</td><td>{info['map']['shape'][0]} × {info['map']['shape'][1]} map, 2 EDTs + bilinear resampling + 4D broadcast</td></tr>
<tr><td>HJSolver setup + kernel compile</td><td>{tm['setup_compile_s']:.2f} s</td><td>HeteroCL graph build (wall − kernel)</td></tr>
<tr><td>HJSolver kernel</td><td>{tm['kernel_s']:.2f} s</td><td>{tm['n_substeps']} sub-steps × 2 RK stages, ≈ {tm['kernel_s']/tm['n_substeps']*1000:.0f} ms per sub-step</td></tr>
<tr><td>HJSolver total (wall)</td><td>{tm['hjsolver_wall_s']:.2f} s</td><td>also stores all {info['tau']['n']} time slices (≈ {up['total']*info['tau']['n']*8/1e6:.0f} MB float64)</td></tr>
</table></div>
<div class="callout">The cost is the same as the analytic-cylinder example (kernel time about 11.7 s on this machine). Solver cost depends on the grid and the dynamics, not on what \\(l(s)\\) looks like, so using a map-based SDF costs nothing extra beyond the {tm['sdf_build_s']*1000:.0f} ms to build it.</div>
<p class="meta">Machine: {env['cpu']}, {env['threads']} threads, WSL 2 Ubuntu 22.04 · Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, HeteroCL {env['heterocl']}.</p>

<h2 id="repro">Reproduce</h2>
<pre><code>conda activate odp
cd ~/optimized_dp
python examples/dubins_4d_avoid_sdf.py          # solve + save to results/dubins_4d_avoid_sdf/
python examples/dubins_4d_avoid_sdf_report.py   # build results/dubins_4d_avoid_sdf/report.html</code></pre>
<p>To use your own map, replace the rasterized <code>occ</code> in <code>build_l()</code> with your occupancy grid, and set <code>map_x</code>/<code>map_y</code> to its cell-center coordinates in meters. <code>sdf_from_occupancy()</code> and the resampling step don't change.</p>
</main>
<script>{TOC_JS}</script>
</body>
</html>
"""
out =os.path.join(RES_DIR, "report.html")
with open(out, "w") as f:
    f.write(html)
print("Report:", os.path.normpath(out), "({:.1f} MB)".format(os.path.getsize(out) / 1e6))
print("stats:", {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
print("slices:", [(r[0], round(r[2], 3)) for r in slice_rows])
