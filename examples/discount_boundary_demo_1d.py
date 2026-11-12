"""
Why grid boundaries can keep the discounted safe set from flattening: a 1D demonstration + HTML report.

1D avoid problem: x_dot = u, |u| <= c, obstacle x <= 0 (l = x), the control maximizes the Hamiltonian
(drives away from the obstacle). Discounted HJ-VI with odp's HJSolver(..., discount=gamma).

On the infinite line the exact value is   V(x, tau) = min( x, e^{-gamma tau} (x + c tau) ) -> 0.
Its decay comes from the far end of the trajectory, outside any finite grid. On a bounded grid the
right edge is an outflow boundary and odp fills the ghost cells by extrapolating V itself, so that
information never arrives. The edge cell then evolves as dV/dtau = c |dV/dx| - gamma V; if
c |dV/dx| >= gamma V there, it can never decrease and the clamp pins it at l (and the pinning
spreads inward). Threshold for l = x at x = X_MAX:  c* = gamma * X_MAX.

Runs (all with accuracy="low", so only the boundary treatment matters):
    sweep  : open edge, c in C_SWEEP           -> pinned iff c >= c*
    open   : open edge, c = C_DEMO > c*         -> safe set stuck at V = l
    margin : l = min(x, X_MAX - WALL - x), c = C_DEMO -> edge inside a constraint, flattens

    python examples/discount_boundary_demo_1d.py
    -> results/discount_boundary_demo_1d/report.html  (+ 4D open-case evidence if
       results/dubins_4d_avoid_discounted/open exists)
"""
import base64
import io
import json
import math
import os
import sys

import heterocl as hcl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from odp.Grid import Grid
from odp.solver import HJSolver

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "results", "discount_boundary_demo_1d")
OPEN_4D = os.path.join(HERE, "..", "results", "dubins_4d_avoid_discounted", "open")

GAMMA = 0.5
T = 20.0
X_MIN, X_MAX, N = -1.0, 5.0, 121
WALL = 0.5           # margin thickness for the "margin" run
C_DEMO = 4.0         # same top speed as the 4D Dubins car
C_SWEEP = [1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 4.0]
ACCURACY = "low"     # monotone scheme, so only the boundary treatment differs between the runs
C = C_DEMO           # speed bound used by Move1D (set before each solve)


class Move1D:
    """x_dot = u, |u| <= C; u maximizes p * u (avoid)."""

    def opt_ctrl(self, t, state, spat_deriv):
        u = hcl.scalar(C, "u")
        with hcl.if_(spat_deriv[0] < 0):
            u[0] = -C
        return (u[0],)

    def opt_dstb(self, t, state, spat_deriv):
        d = hcl.scalar(0, "d")
        return (d[0],)

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        x_dot[0] = uOpt[0] + dOpt[0]
        return (x_dot[0],)


def solve(l, g, tau):
    real = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        return HJSolver(Move1D(), g, l, tau, {"TargetSetMode": "minVWithV0"}, saveAllTimeSteps=True,
                        accuracy=ACCURACY, discount=GAMMA)
    finally:
        sys.stdout.close()
        sys.stdout = real


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    g = Grid(np.array([X_MIN]), np.array([X_MAX]), 1, np.array([N]))
    x = g.grid_points[0]
    dx = float(g.dx[0])
    tau = np.arange(0, T + 1e-5, 0.5)
    n_t = len(tau)
    c_star = GAMMA * X_MAX               # threshold: c |dl/dx| = gamma l at the edge, with l = x
    safe = x > 0

    def at(R, t):
        return R[:, n_t - 1 - int(round(t / 0.5))]

    def exact_open(c, t):
        return np.minimum(x, math.exp(-GAMMA * t) * (x + c * t))

    # threshold sweep (open edge)
    sweep = []
    for c in C_SWEEP:
        C = c
        V = solve(x.copy(), g, tau)[:, 0]
        sweep.append(dict(c=c, edge=float(V[-1]), exact_edge=float(exact_open(c, T)[-1]),
                          pinned=int(np.sum(safe & (np.abs(V - x) < 1e-6))), n_safe=int(safe.sum())))
        print(sweep[-1], flush=True)

    # demo runs at the 4D car's top speed
    C = C_DEMO
    l_open = x.copy()
    l_margin = np.minimum(x, X_MAX - WALL - x)
    R_open = solve(l_open, g, tau)
    R_margin = solve(l_margin, g, tau)
    stats = dict(
        gamma=GAMMA, T=T, dx=dx, accuracy=ACCURACY, c_demo=C_DEMO, c_star=c_star,
        open_max_safe=float(at(R_open, T)[safe].max()),
        open_exact_max_safe=float(exact_open(C_DEMO, T)[safe].max()),
        margin_max_safe=float(at(R_margin, T)[at(R_margin, T) > 0].max()),
        margin_bound=float(math.exp(-GAMMA * T) * l_margin.max()),
        sweep=sweep)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "V_open.npy"), R_open)
    np.save(os.path.join(OUT_DIR, "V_margin.npy"), R_margin)
    json.dump(stats, open(os.path.join(OUT_DIR, "stats.json"), "w"), indent=1)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
    t_list = [0, 1, 2, 5, 20]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(t_list)))

    # Fig 1: open edge at c = 4 vs exact; and threshold sweep
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for c_, t in zip(cols, t_list):
        axs[0].plot(x, exact_open(C_DEMO, t), color=c_, lw=1.0, ls="--")
        axs[0].plot(x, at(R_open, t), color=c_, lw=1.8, alpha=0.85, label="τ = {} s".format(t))
    axs[0].plot([], [], "k--", lw=1.0, label="exact (infinite line)")
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("V(x, τ)")
    axs[0].set_title("Open edge, c = {:g}: odp (solid, all τ on top of l) vs exact (dashed)".format(C_DEMO))
    axs[0].legend(fontsize=7.5, loc="upper left")
    cs = [s["c"] for s in sweep]
    axs[1].semilogy(cs, [s["edge"] for s in sweep], "o-", color="#c0392b", lw=1.8, label="odp, open edge")
    axs[1].semilogy(cs, [s["exact_edge"] for s in sweep], "s--", color="0.4", lw=1.2, label="exact (infinite line)")
    axs[1].axvline(c_star, color="#2f6f5e", lw=1.2, ls=":")
    axs[1].text(c_star, 1e-2, "  c* = γ·l(edge) = {:g}".format(c_star), color="#2f6f5e", fontsize=8)
    axs[1].set_xlabel("speed bound c")
    axs[1].set_ylabel("V at the edge x = {:g}, τ = {:g}".format(X_MAX, T))
    axs[1].set_title("Threshold: pinned at l = {:g} iff c ≥ c*".format(X_MAX))
    axs[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig1 = png(fig)

    # Fig 2: ghost-cell schematic at the right edge (open run)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    k = 5
    Vf = at(R_open, T)
    xs_in, vs_in = x[-k:], Vf[-k:]
    s = abs(vs_in[-1] - vs_in[-2]) * np.sign(vs_in[-1])
    gx = [X_MAX + dx, X_MAX + 2 * dx]
    gv = [vs_in[-1] + s, vs_in[-1] + 2 * s]
    ax.plot(xs_in, vs_in, "o-", color="#2f6f5e", label="grid values V_i (= l, pinned)")
    ax.plot([X_MAX] + gx, [vs_in[-1]] + gv, "s--", color="#c0392b", mfc="white", label="ghost values built by odp")
    ax.plot(xs_in[-1] + np.array([0, 2 * dx]), [math.exp(-GAMMA * T) * (X_MAX + C_DEMO * T)] * 2, color="0.4", lw=1.0, ls=":",
            label="what the exact solution needs out there (≈ {:.0e})".format(math.exp(-GAMMA * T) * (X_MAX + C_DEMO * T)))
    ax.axvline(X_MAX, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("x")
    ax.set_ylabel("V at τ = T")
    ax.set_title("Right edge: the ghost continues V's own slope, never a discounted value")
    ax.legend(fontsize=7.5, loc="center left")
    fig2 = png(fig)

    # Fig 3: margin run
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for c_, t in zip(cols, t_list):
        axs[0].plot(x, at(R_margin, t), color=c_, lw=1.8, label="τ = {} s".format(t))
    axs[0].plot(x, l_margin, color="0.5", lw=1.0, ls="--", label="l = min(x, {:g} − x)".format(X_MAX - WALL))
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].axvspan(X_MAX - WALL, X_MAX, color="0.85", zorder=0)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("V(x, τ)")
    axs[0].set_title("Margin, c = {:g}: grid edge inside a constraint (grey, l < 0)".format(C_DEMO))
    axs[0].legend(fontsize=7.5, loc="upper right")
    mx_open = [at(R_open, t)[safe].max() for t in tau]
    mx_margin = [max(at(R_margin, t)[at(R_margin, t) > 0].max(), 1e-12) for t in tau]
    axs[1].semilogy(tau, mx_open, color="#c0392b", lw=1.8, label="open edge")
    axs[1].semilogy(tau, mx_margin, color="#2f6f5e", lw=1.8, label="margin")
    axs[1].semilogy(tau, np.exp(-GAMMA * tau) * l_margin.max(), color="#2f6f5e", lw=1.0, ls=":",
                    label="bound e^{−γτ} max l (margin)")
    axs[1].set_xlabel("time-to-go τ")
    axs[1].set_ylabel("max V over the safe set")
    axs[1].set_title("Does the safe set flatten? (c = {:g})".format(C_DEMO))
    axs[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig3 = png(fig)

    # Fig 4: 4D evidence from the existing open-case run (no recomputation)
    d4_html = ""
    if os.path.isdir(OPEN_4D):
        info = json.load(open(os.path.join(OPEN_4D, "run_info.json")))
        pts = info["grid"]["pts"]
        I, J, K = np.meshgrid(np.arange(pts[0]), np.arange(pts[1]), np.arange(pts[2]), indexing="ij")
        bd = np.minimum.reduce([I, pts[0] - 1 - I, J, pts[1] - 1 - J])[..., None]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        d4 = {}
        for r, col in zip([info["runs"][0], info["runs"][-1]], ["#5b6470", "#c0392b"]):
            Vf4 = np.load(os.path.join(OPEN_4D, "V_final_{}.npy".format(r["tag"])))
            bdf = np.broadcast_to(bd, Vf4.shape)
            med = [float(np.median(Vf4[(bdf == dd) & (Vf4 > 0)])) for dd in range(12)]
            ax.plot(range(12), med, "o-", color=col, lw=1.6, ms=3.5, label="γ = {}".format(r["gamma"]))
            d4[r["gamma"]] = dict(max_pos=float(Vf4[Vf4 > 0].max()), median_edge=med[0], median_deep=med[-1])
        ax.set_xlabel("distance to nearest x / y grid edge (cells)")
        ax.set_ylabel("median V on the safe set at T")
        ax.set_title("4D Dubins, open grid, T = {:g} s".format(info["tau"]["lookback_length"]))
        ax.legend(fontsize=8)
        fig4 = png(fig)
        g0, g9 = min(d4), max(d4)
        T4 = info["tau"]["lookback_length"]
        d4_html = f"""
<h2 id="evidence4d">4. The same effect in the 4D Dubins example</h2>
<p>These are the existing open-domain results (original grid, no margins), recomputed from disk. With γ = {g9} the exact safe values are at most about \\(e^{{-\\gamma T}}(l + 4T) \\approx {math.exp(-g9 * T4) * (3.44 + 4 * T4):.0e}\\). Instead, the maximum is <b>{d4[g9]['max_pos']:.2f}</b>, exactly \\(l\\) at the domain corner (pinned), and the median safe value next to the x/y edges is {d4[g9]['median_edge']:.2f}. The same threshold explains it: at the corner \\(c = v_{{max}} = 4\\), \\(|\\nabla l| = 1\\) and \\(\\gamma l = {g9}\\cdot 3.44 = {g9 * 3.44:.1f} < 4\\).</p>
<figure><img src="{fig4}" alt="4D open case: median V vs distance to the x/y grid edge">
<figcaption><b>Figure 4.</b> 4D open case at T: median V on the safe set against distance to the nearest x/y grid edge. The upward trend toward the edges is just \\(l\\) growing away from the obstacle (γ = 0 shows it too). What matters is the level: with γ = {g9} every safe value should be ≲ {math.exp(-g9 * T4) * (3.44 + 4 * T4):.0e}, but it's {d4[g9]['median_deep']:.2f}–{d4[g9]['median_edge']:.2f}. The discount barely acts.</figcaption></figure>
<p>Coarse 4D checks (30×30×12×18, γ = 0.9, T = 15 s) ruled out the other suspects: the median safe value inside the domain was 0.70 with <code>accuracy="medium"</code>, 0.68 with <code>"low"</code>, and 0.69 with <code>"low"</code> plus speed saturated to [0, 4] (no v-edge outflow). So the cause is the x/y edges, not the scheme order or the v edges. With every edge inside a margin (the arena case in <code>DISCOUNTED_AVOID_REPORT.html</code>), the safe set flattens.</p>
"""

    rows = "\n".join("<tr><td>{c:g}</td><td>{edge:.3g}</td><td>{exact_edge:.1e}</td><td>{pinned} / {n_safe}</td><td>{verdict}</td></tr>".format(
        verdict="<b>pinned</b>" if s["c"] >= c_star else "decays", **s) for s in sweep)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boundary Source Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}]}});"></script>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --border:#e3e0d8; --code-bg:#f3f1ec;
        --accent:#2f6f5e; --accent-bg:#e6f2ee; --warn:#9a5b00; --warn-bg:#fff6e5; color-scheme: light; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:980px; margin:0 auto; padding:32px 16px 80px; }}
h1 {{ font-size:1.7rem; margin:0 0 4px; }}
h2 {{ font-size:1.22rem; margin:40px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; scroll-margin-top:16px; }}
.meta {{ color:var(--muted); font-size:.9rem; }}
code {{ font-family:ui-monospace,Consolas,monospace; background:var(--code-bg); padding:0 4px; border-radius:3px; font-size:.92em; }}
pre {{ background:var(--code-bg); padding:10px 12px; border-radius:6px; overflow-x:auto; font:12.5px/1.45 ui-monospace,Consolas,monospace; }}
figure {{ margin:16px 0 22px; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ color:var(--muted); font-size:.88rem; margin-top:8px; }}
figcaption b {{ color:var(--fg); }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
th, td {{ text-align:left; border-bottom:1px solid var(--border); padding:6px 8px; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }}
.callout {{ background:var(--accent-bg); border-left:3px solid var(--accent); padding:10px 14px; border-radius:4px; margin:12px 0; }}
.warn {{ background:var(--warn-bg); border-left:3px solid var(--warn); padding:10px 14px; border-radius:4px; margin:12px 0; }}
ol.steps > li {{ margin-bottom:10px; }}
</style>
</head>
<body>
<main>
<h1>Boundary Source Report</h1>
<p class="meta">Reason 2 for the missing flat-zero safe region: outflow grid edges · branch <code>discounted-avoid</code> (uncommitted) · generated by <code>examples/discount_boundary_demo_1d.py</code></p>

<div class="callout"><b>In short:</b> the discounted value decays because of what happens <em>far along</em> each trajectory, which for a safe state is usually outside the grid. odp replaces the outside with ghost cells extrapolated from V itself, so that decay never arrives. An outflow edge cell then evolves as \\(\\partial_\\tau V = c\\,|\\nabla V| - \\gamma V\\). If \\(c\\,|\\nabla V| \\ge \\gamma V\\) there, it can't decrease, gets pinned at \\(l\\), and holds the neighboring safe values up with it.</div>

<h2 id="exact">1. Where the decay comes from</h2>
$$V(x,\\tau) = \\max_{{u(\\cdot)}}\\ \\min_{{s\\in[0,\\tau]}}\\ e^{{-\\gamma s}}\\, l\\big(x(s)\\big)$$
<p>For a safe state the minimum is eventually attained at the <b>end</b> of the trajectory, \\(s = \\tau\\), where the factor \\(e^{{-\\gamma\\tau}}\\) has made the term small. In the 1D demo (\\(\\dot x = u,\\ |u| \\le c\\), obstacle \\(x \\le 0\\), \\(l = x\\)) the best strategy is to drive away, and</p>
$$V(x,\\tau) = \\min\\big(x,\\ e^{{-\\gamma\\tau}}\\,(x + c\\tau)\\big).$$
<p>The second term uses \\(l\\) at \\(x + c\\tau\\): with c = {C_DEMO:g} and τ = {T:g} that's at x ≈ {X_MAX + C_DEMO * T:.0f}, far outside a grid that ends at x = {X_MAX:g}. In the HJ PDE this information travels along characteristics that enter the grid through the outflow edge.</p>

<h2 id="mechanism">2. What happens at an outflow edge in odp</h2>
<ol class="steps">
<li><b>Ghost cells copy V's own trend.</b> <code>odp/spatialDerivatives/*/…</code> fills cells beyond a non-periodic edge by extrapolating V linearly away from zero (1st order shown; 2nd order is the same idea):
<pre>right_boundary = V[N] + |V[N] - V[N-1]| * sign(V[N])      # first_orderENO1D.py, i = N
right_deriv    = (right_boundary - V[N]) / dx               # = |V[N] - V[N-1]| / dx</pre>
The outside never "knows" about \\(e^{{-\\gamma\\tau}}\\); it's rebuilt from the current V at every step.</li>
<li><b>Edge equation.</b> Moving outward at speed c, the upwind Hamiltonian at the edge is \\(c\\,p^+ = c\\,|\\nabla V|\\), so
$$\\partial_\\tau V_N = c\\,|\\nabla V|_N - \\gamma V_N .$$</li>
<li><b>Threshold.</b> At τ = 0, V = l. If \\(c\\,|\\nabla l| \\ge \\gamma\\, l\\) at the edge, the right side is ≥ 0, V can't go down, and the clamp \\(V \\le l\\) keeps it at \\(l\\). Then the slope stays \\(|\\nabla l|\\), the ghost reproduces it, and the condition holds forever: <b>pinned</b>. Interior cells with \\(c|\\nabla l| \\ge \\gamma l\\) are pinned the same way, because their upwind neighbor never drops.</li>
<li><b>Below the threshold</b> (\\(c|\\nabla V| < \\gamma V\\)) the edge decays. In the 1D linear case the extrapolation even happens to reproduce the exact solution, which stays linear in x.</li>
</ol>
<p>So the critical quantity is the ratio of the edge value to its outward slope, compared with \\(c/\\gamma\\):</p>
$$\\text{{pinned}} \\iff \\frac{{l}}{{|\\nabla l|}}\\Big|_{{\\text{{outflow edge}}}} \\le \\frac{{c}}{{\\gamma}}.$$
<p>Fast systems (large c) and gentle discounts (small γ) make it easy to hit. The 4D Dubins car reaches c = 4 m/s.</p>

<h2 id="demo">3. 1D demonstration</h2>
<p>γ = {GAMMA}, T = {T:g}, grid \\([{X_MIN:g}, {X_MAX:g}]\\) with {N} points (Δx = {dx:.3f}), <code>accuracy="{ACCURACY}"</code>. Threshold for \\(l = x\\) at the edge: \\(c^* = \\gamma\\cdot{X_MAX:g} = {c_star:g}\\).</p>
<figure><img src="{fig1}" alt="Open edge vs exact, and threshold sweep">
<figcaption><b>Figure 1.</b> <b>Left:</b> open edge with c = {C_DEMO:g} (above c*). The exact solution (dashed) decays toward 0, while odp (solid) stays on \\(V = l = x\\) at every τ: all {sweep[-1]['n_safe']} safe cells are pinned. <b>Right:</b> V at the edge after τ = {T:g} against the speed bound. Below c* odp matches the exact value; from c* on it jumps to \\(l = {X_MAX:g}\\).</figcaption></figure>
<table style="max-width:720px">
<tr><th>c</th><th>odp V at edge, τ = {T:g}</th><th>exact</th><th>safe cells pinned at l</th><th></th></tr>
{rows}
</table>
<figure><img src="{fig2}" alt="Ghost cells at the right edge">
<figcaption><b>Figure 2.</b> The last grid cells of the open run (c = {C_DEMO:g}) and the two ghost values odp builds from them. They continue V's own slope. The exact solution would need the outside to be ≈ {math.exp(-GAMMA * T) * (X_MAX + C_DEMO * T):.0e} (dotted), which no extrapolation of V can produce.</figcaption></figure>
<figure><img src="{fig3}" alt="Margin run">
<figcaption><b>Figure 3.</b> <b>Margin</b> with c = {C_DEMO:g}: \\(l = \\min(x,\\ {X_MAX - WALL:g} - x)\\), so the outer {WALL:g} of the grid is a constraint with \\(l < 0\\). Safe trajectories no longer leave the grid, the edge holds negative values, and the ghosts stay negative. The safe values decay to 0 under the bound \\(e^{{-\\gamma\\tau}}\\max l\\): at τ = {T:g}, max {stats['margin_max_safe']:.1e} vs bound {stats['margin_bound']:.1e}. Right: max safe value over time, open (pinned at {stats['open_max_safe']:.0f}) vs margin.</figcaption></figure>
{d4_html}
<h2 id="fix">{"5" if d4_html else "4"}. Consequences and fix</h2>
<ul>
<li><b>Only the flatness is affected.</b> The pinned cells are safe states (V &gt; 0), so the unsafe set is right; the safe values are just not discounted.</li>
<li><b>Without discount nothing is wrong:</b> with γ = 0 the true value of a safe state is positive anyway, and \\(c|\\nabla V| \\ge 0\\) holding V at a positive level is correct.</li>
<li><b>Fix:</b> put every non-periodic edge inside a margin where \\(l < 0\\), so no safe trajectory leaves the grid (the arena setup: walls of thickness \\(w > 0\\), an extended speed grid with out-of-range speeds unsafe).</li>
<li>Alternatively make the domain large enough that \\(l/|\\nabla l| > c/\\gamma\\) at every outflow edge, but in practice that means a far bigger grid.</li>
</ul>

<h2 id="repro">Reproduce</h2>
<pre><code>conda activate odp
cd ~/optimized_dp
git switch discounted-avoid
python examples/discount_boundary_demo_1d.py   # ~30 s; writes results/discount_boundary_demo_1d/report.html</code></pre>
<p class="meta">This 1D demo doubles as a quick regression check: the open run at c = {C_DEMO:g} is pinned with odp's current edge treatment, and the margin run must flatten.</p>
</main>
</body>
</html>
"""
    out = os.path.join(OUT_DIR, "report.html")
    with open(out, "w") as f:
        f.write(html)
    print(json.dumps({k: v for k, v in stats.items() if k != "sweep"}, indent=1))
    print("Report:", os.path.normpath(out), "({:.1f} MB)".format(os.path.getsize(out) / 1e6))
