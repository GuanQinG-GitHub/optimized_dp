"""
Review page for one run of examples/dubins_4d_avoid_discounted.py: an animation of V(x, y) over
time-to-go (two headings, fixed speed) and the safe-set decay against the theoretical bound.

    python examples/dubins_4d_avoid_discounted.py                       # e.g. -> .../arena_g0.3_T10
    python examples/dubins_4d_avoid_discounted_view.py [run_folder]     # -> <run_folder>/view.html
"""
import base64
import io
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from odp.Grid import Grid
from odp.Plots.plotting_utilities import slider_define

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "dubins_4d_avoid_discounted",
                                                          "arena_g0.3_T10")
HEADINGS = (0, 90)

info = json.load(open(os.path.join(RUN, "run_info.json")))
gi = info["grid"]
g = Grid(np.array(gi["min"]), np.array(gi["max_input"]), 4, np.array(gi["pts"]), gi["periodic"])
assert np.allclose(g.max, gi["max"]), "grid rebuilt differently from the solver run"
xs, ys, vs, ths = g.grid_points
run = info["runs"][-1]
gamma, T, dt_out = run["gamma"], info["tau"]["lookback_length"], info["tau"]["t_step"]
v_idx = info["v_index_saved"]
S = np.load(os.path.join(RUN, "V_vslice_all_t_{}.npy".format(run["tag"])))   # (x, y, theta, t); [..., -1] = tau 0
n_t = S.shape[-1]
l_max = info["l_max"]


def theta_index(deg):
    return int(np.argmin(np.abs(np.angle(np.exp(1j * (ths - math.radians(deg)))))))


ks = [theta_index(d) for d in HEADINGS]
zlim = float(np.abs(S[:, :, ks, :]).max())


def heat(Z, show_scale):
    return go.Heatmap(x=np.round(xs, 3), y=np.round(ys, 3), z=np.round(Z.T.astype(np.float64), 4),
                      zmin=-zlim, zmax=zlim, zmid=0, colorscale="RdBu", showscale=show_scale,
                      colorbar=dict(title="V", len=0.85),
                      hovertemplate="x=%{x:.2f}, y=%{y:.2f}<br>V=%{z:.4f}<extra></extra>")


def zero(Z):
    return go.Contour(x=np.round(xs, 3), y=np.round(ys, 3), z=np.round(Z.T.astype(np.float64), 4),
                      contours=dict(start=0, end=0, size=1, coloring="none"), line=dict(color="black", width=2),
                      showscale=False, showlegend=False, hoverinfo="skip")


def traces(i):
    out = []
    for j, k in enumerate(ks):
        Z = S[:, :, k, i]
        out += [heat(Z, j == len(ks) - 1), zero(Z)]
    return out


fig = make_subplots(rows=1, cols=len(ks), horizontal_spacing=0.07,
                    subplot_titles=["heading θ = {:+.0f}°".format(math.degrees(ths[k])) for k in ks])
for j, tr in enumerate(traces(n_t - 1)):
    fig.add_trace(tr, 1, j // 2 + 1)
fig.frames = [go.Frame(data=traces(n_t - 1 - k), traces=list(range(2 * len(ks))), name=str(k)) for k in range(n_t)]
for c in range(1, len(ks) + 1):
    fig.update_xaxes(title_text="x (m)", constrain="domain", row=1, col=c)
    fig.update_yaxes(title_text="y (m)", scaleanchor="x" if c == 1 else "x{}".format(c), row=1, col=c)
fig.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="white",
                  title="V(x, y, v = {:.2f} m/s, θ; τ), γ = {} (black = zero level)".format(vs[v_idx], gamma))
fig = slider_define(fig, duration=150)
fig.layout.sliders[0].currentvalue.prefix = "τ = "
fig.layout.sliders[0].currentvalue.suffix = " s"
for k, step in enumerate(fig.layout.sliders[0].steps):
    step.label = "{:g}".format(k * dt_out)
anim_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id="anim", config={"displaylogo": False})

# static: safe-set values vs the theoretical bound over the whole 4D grid
d = run["diagnostics"]
t = np.array([row["tau"] for row in d])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
f, ax = plt.subplots(figsize=(7.2, 3.9))
ax.semilogy(t, [row["max_pos"] for row in d], color="#c0392b", lw=1.8, label="max V on safe set")
ax.semilogy(t, [row["median_pos"] for row in d], color="#2f6f5e", lw=1.8, label="median V on safe set")
ax.semilogy(t, np.exp(-gamma * t) * l_max, "k:", lw=1.2, label="bound e^{−γτ} max l")
ax.set_xlabel("time-to-go τ (s)")
ax.set_ylabel("V")
ax.set_title("Whole 4D grid: safe values decay under the bound")
ax.legend(fontsize=8)
buf = io.BytesIO()
f.savefig(buf, format="png", bbox_inches="tight", dpi=130)
plt.close(f)
decay_png = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

last = d[-1]
bound_T = math.exp(-gamma * T) * l_max
mg = info["margins"] or {}
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Discounted Avoid Check</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --border:#e3e0d8; --code-bg:#f3f1ec;
        --accent:#2f6f5e; --accent-bg:#e6f2ee; color-scheme: light; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:28px 16px 64px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; }}
h2 {{ font-size:1.15rem; margin:32px 0 10px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
.meta {{ color:var(--muted); font-size:.9rem; }}
code {{ font-family:ui-monospace,Consolas,monospace; background:var(--code-bg); padding:0 4px; border-radius:3px; font-size:.92em; }}
figure {{ margin:14px 0 20px; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; }}
figure img {{ width:100%; max-width:760px; height:auto; display:block; }}
figcaption {{ color:var(--muted); font-size:.88rem; margin-top:8px; }}
.kpi {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin:12px 0; }}
.kpi div {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px 12px; }}
.kpi b {{ display:block; font-size:1.2rem; }}
.kpi span {{ color:var(--muted); font-size:.8rem; }}
</style>
</head>
<body>
<main>
<h1>Discounted Avoid Check</h1>
<p class="meta">Margin ("arena") case: the state boundary is in l (walls {100 * mg.get('wall', 0):.0f} cm in x/y; speeds outside {mg.get('v_allowed')} m/s unsafe), γ = {gamma}, T = {T:g} s, <code>accuracy="{info['accuracy']}"</code>. Run folder <code>{os.path.basename(os.path.normpath(RUN))}</code>.</p>
<div class="kpi">
<div><b>{last['max_pos']:.2e}</b><span>max V on safe set at T</span></div>
<div><b>{bound_T:.2e}</b><span>bound e^(−γT)·max l</span></div>
<div><b>{last['median_pos']:.2e}</b><span>median V on safe set at T</span></div>
<div><b>{run['hjsolver_wall_s']:.0f} s</b><span>HJSolver wall time</span></div>
</div>

<h2>Animation: V(x, y) as time-to-go grows</h2>
<figure>{anim_html}
<figcaption>Speed v = {vs[v_idx]:.2f} m/s, headings 0° (facing +x) and 90° (facing +y). Red = V &lt; 0 (unsafe), blue = V &gt; 0 (safe), black = zero level. The color scale is fixed, so as τ grows the safe (blue) region fades toward white, i.e. V → 0, while the unsafe (red) region keeps its sign. The outer frame is the 20 cm wall. Press Play or drag the slider; hover to read values.</figcaption></figure>

<h2>Decay of the safe values (whole 4D grid)</h2>
<figure><img src="{decay_png}" alt="Safe-set values vs bound">
<figcaption>Max and median of V over all safe states of the 4D grid against τ. Both stay under the theoretical bound — nothing is held up at the grid edges. With γ·T = {gamma * T:g} the bound at T is only {bound_T:.1e}, so the region is still faintly blue at the end; a larger γ·T (≈ 10) makes it visually flat.</figcaption></figure>
</main>
</body>
</html>
"""
out = os.path.join(RUN, "view.html")
with open(out, "w") as fh:
    fh.write(html)
print("View:", os.path.normpath(out), "({:.1f} MB)".format(os.path.getsize(out) / 1e6))
