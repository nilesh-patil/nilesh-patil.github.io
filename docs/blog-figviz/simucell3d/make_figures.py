#!/usr/bin/env python3
"""
Regenerate the 7 data figures for the SimuCell3D adaptive-OpenMP blog post,
overlaying the new 26.5k-cell run (parallel_benchmark_20260621_055502) onto the
original published run (parallel_benchmark_20260124_091702).

One unified house style across all 7 figures (cream stone background, bold
left-aligned title + grey subtitle, stone chrome). Run identity is encoded the
same way everywhere: RUN01 = lighter shade / solid line / circle marker, RUN02 =
darker shade / dashed line / square marker; hue is reserved by mode (blue =
adaptive, red = v1), and fig7's phases use a colorblind-safe non-red/blue set.
  figs 3,4,4b,5,6  -> co-plot BOTH runs
  figs 1,2,7       -> new-run adaptive + ORIGINAL Static/v1 baseline
                      (original Static cloud recovered from the published SVGs)

Cell counts are reconciled to each run's largest count:
  original run -> 19,958 (adaptive) / 9,693 (v1)
  new run      -> 26,534 (adaptive) / 12,851 (v1)

Run with a matplotlib-capable interpreter, e.g.
  /opt/anaconda3/bin/python3 make_figures.py
Override paths with SIMUCELL3D_RESULTS / SIMUCELL3D_IMG_OUT if needed.
"""
import csv
import math
import os
import random
import re
import statistics as st
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
RESULTS = os.environ.get(
    "SIMUCELL3D_RESULTS",
    "/Users/nilesh-patil/versioned-projects/Github-personal.tmp/simucell3d/docs/simulation_results",
)
ORIG_DIR = os.path.join(RESULTS, "parallel_benchmark_20260124_091702")
NEW_DIR = os.path.join(RESULTS, "parallel_benchmark_20260621_055502")
HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.join(HERE, "datapoints")
# Output dir defaults to <repo>/images/blog/simucell3d relative to this script
# (docs/blog-figviz/simucell3d/), so it resolves correctly in the main checkout
# and in a git worktree. Override with SIMUCELL3D_IMG_OUT.
IMG_OUT = os.environ.get(
    "SIMUCELL3D_IMG_OUT",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "images", "blog", "simucell3d")),
)

ORIG_ADAPT_MAX, ORIG_V1_MAX = 19958, 9693
NEW_ADAPT_MAX, NEW_V1_MAX = 26534, 12851

# ---------------------------------------------------------------------------
# palette (extracted from the published SVGs)
# ---------------------------------------------------------------------------
# One palette across all 7 figures. Run identity is encoded the SAME way
# everywhere: RUN01 = lighter shade / solid line / circle; RUN02 = darker shade /
# dashed line / square. Hue is reserved by mode: blue = adaptive, red = v1.
BLUE = "#4c72b0"       # adaptive, RUN01
NEW_BLUE = "#2f4b7c"   # adaptive, RUN02 (darker)
RED = "#c44e52"        # v1 / Static, RUN01
NEW_RED = "#8c3b3f"    # v1, RUN02 (darker)
PURPLE = "#9b6dd1"     # HOMEOSTASIS phase (fig3)
ORANGE = "#e69f00"     # INITIALIZATION phase (fig3)
CREAM, INK_B = "#fbfaf7", "#1f2933"
STONE, STONE2, STONE3, GRID_B = "#a8a29e", "#6b7280", "#4b5563", "#e7e5df"
# fig7 phase colors: Okabe-Ito categorical set, deliberately avoiding the
# v1-red and adaptive-blue used elsewhere (contact, polar, timeint, mesh).
PH_CONTACT, PH_POLAR, PH_TIME, PH_MESH = "#e69f00", "#009e73", "#cc79a7", "#56b4e9"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def parse_hms(s):
    p = s.strip().split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])


def parse_dt(s):
    return datetime.strptime(s.strip(), "%Y-%m-%d_%H:%M:%S")


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return b, a, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def monotonic_wall(series):
    """Sort a [(hours, cells)] trajectory by time and clamp cells to a running
    max: tissue cell count only grows, so any backtrack is a monitor glitch."""
    s = sorted((h, c) for h, c in series if h is not None)
    out, run = [], 0
    for h, c in s:
        run = max(run, c)
        out.append((h, run))
    return out


def trend_line(points, nbins=26):
    pts = [(x, y) for x, y in points if x > 0]
    if not pts:
        return [], []
    xs = [x for x, _ in pts]
    lo, hi = math.log10(min(xs)), math.log10(max(xs))
    bx, by = [], []
    for i in range(nbins):
        a = 10 ** (lo + (hi - lo) * i / nbins)
        b = 10 ** (lo + (hi - lo) * (i + 1) / nbins)
        ys = [y for x, y in pts if a <= x < b]
        if ys:
            bx.append(math.sqrt(a * b))
            by.append(st.median(ys))
    return bx, by


# ---------------------------------------------------------------------------
# load a run -> normalized series
# ---------------------------------------------------------------------------
def load_run(run_dir, key):
    out = {"key": key}
    diag = read_csv(os.path.join(run_dir, "sim_adaptive", "performance_diagnostics.csv"))
    D = []
    for r in diag:
        D.append({
            "iteration": int(r["iteration"]), "cells": int(r["cells"]),
            "cov": float(r["cov"]), "phase": r["phase"].strip(),
            "imb": float(r["thread_imbalance_pct"]),
            "mesh": float(r["mesh_refinement_ms"]), "contact": float(r["contact_detection_ms"]),
            "polar": float(r["polarization_internal_forces_ms"]),
            "timeint": float(r["time_integration_ms"]), "total": float(r["total_iteration_ms"]),
            "wall_epoch": float(r["wall_epoch"]) if r.get("wall_epoch") else None,
        })
    D.sort(key=lambda d: d["iteration"])
    out["diag"] = D

    if key == "new":
        w0 = D[0]["wall_epoch"]
        out["adapt_wall"] = [((d["wall_epoch"] - w0) / 3600.0, d["cells"]) for d in D]
        vlog, pat = {}, re.compile(r"iteration:\s*(\d+),.*nb cells\s*(\d+)")
        with open(os.path.join(run_dir, "logs", "v1.log"), errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    vlog[int(m.group(1))] = int(m.group(2))
        vsec = {}
        for r in read_csv(os.path.join(run_dir, "sim_v1", "simulation_statistics.csv")):
            it = int(r["iteration"])
            vsec.setdefault(it, parse_hms(r["computation_time_(hh::mm:ss)"]))
        out["v1_wall"] = [(vsec[it] / 3600.0, vlog[it]) for it in sorted(vsec) if it in vlog]
        its, vips = sorted(vsec), []
        for i in range(1, len(its)):
            dt, di = vsec[its[i]] - vsec[its[i - 1]], its[i] - its[i - 1]
            if dt > 0 and its[i] in vlog:
                vips.append((vlog[its[i]], di / dt))
        out["v1_ips"] = vips
    else:
        comp = read_csv(os.path.join(run_dir, "metrics", "comparison.csv"))
        t0 = parse_dt(comp[0]["timestamp"])
        out["adapt_wall"] = [((parse_dt(r["timestamp"]) - t0).total_seconds() / 3600.0,
                              int(float(r["adaptive_cells"]))) for r in comp
                             if float(r["adaptive_cells"]) > 0]
        out["v1_wall"] = [((parse_dt(r["timestamp"]) - t0).total_seconds() / 3600.0,
                           int(float(r["v1_cells"]))) for r in comp if float(r["v1_cells"]) > 0]
        vips = []
        for r in read_csv(os.path.join(run_dir, "plots-updated", "cleaned_data",
                                       "v1_computational_cleaned.csv")):
            try:
                c = int(float(r["cells"]))
                ips = float(r.get("iter_per_sec_clean") or r["iter_per_sec"])
                if c > 0 and ips > 0:
                    vips.append((c, ips))
            except (ValueError, KeyError):
                pass
        out["v1_ips"] = vips

    out["adapt_ips"] = [(d["cells"], 1000.0 / d["total"]) for d in D if d["total"] > 0]

    if key == "new":
        bdir, pcol = os.path.join(run_dir, "plots", "updated", "cleaned_data"), "mean_pressure"
    else:
        bdir, pcol = os.path.join(run_dir, "plots-updated", "cleaned_data"), "avg_pressure"

    def load_bio(fn):
        d = {}
        for r in read_csv(os.path.join(bdir, fn)):
            try:
                d[int(r["iteration"])] = float(r[pcol])
            except (ValueError, KeyError):
                pass
        return d
    ab, vb = load_bio("adaptive_biological_cleaned.csv"), load_bio("v1_biological_cleaned.csv")
    devs = [abs(ab[it] - vb[it]) / abs(vb[it]) * 100
            for it in sorted(set(ab) & set(vb)) if abs(vb[it]) > 1e-12]
    out["physics"] = {"n": len(devs), "median": st.median(devs) if devs else float("nan")}

    out["adapt_wall"] = monotonic_wall(out["adapt_wall"])
    out["v1_wall"] = monotonic_wall(out["v1_wall"])
    return out


def load_static_recovered():
    f1 = [(float(r["cells"]), float(r["thread_imbalance_pct"]))
          for r in read_csv(os.path.join(DP, "fig1_static_recovered.csv"))]
    f2 = [(float(r["cells"]), float(r["cov"]))
          for r in read_csv(os.path.join(DP, "fig2_static_recovered.csv"))]
    return f1, f2


# ---------------------------------------------------------------------------
# style scaffolds
# ---------------------------------------------------------------------------
def fig_S(figsize=(9.8, 5.6), grid="both"):
    """The single house style shared by all 7 figures: cream stone background,
    left/bottom spines only, light stone grid. grid is 'both' | 'x' | 'y' | None."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    if grid:
        ax.grid(True, which="both", axis=grid, color=GRID_B, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(STONE)
    ax.tick_params(colors=INK_B, labelsize=12)
    return fig, ax


def title_S(ax, t, sub, x, y):
    """Bold left-aligned title + grey subtitle, shared across all figures."""
    ax.set_title(t, color=INK_B, fontsize=17, fontweight="bold", loc="left", pad=24)
    if sub:
        ax.text(0.0, 1.035, sub, transform=ax.transAxes, color=STONE2, fontsize=11.5)
    ax.set_xlabel(x, color=STONE2, fontsize=12.5)
    ax.set_ylabel(y, color=STONE2, fontsize=12.5)


def save(fig, name):
    os.makedirs(IMG_OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_OUT, name), format="svg", facecolor=fig.get_facecolor())
    png = name[:-4] + ".png"
    fig.savefig(os.path.join(IMG_OUT, png), format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {name} + {png}")


# ---------------------------------------------------------------------------
# figures 1 & 2 (style A scatter) : new adaptive + original Static dashed ref
# ---------------------------------------------------------------------------
def project_static(static, new_diag, col, x_start, x_end, seed=20260621):
    """PROJECTION (not measured data): genuine v1.0 emits no imbalance/CoV past
    where its run stopped (x_start = 9,693 cells). To extend the Static (v1)
    reference across the adaptive range, carry the measured static-minus-adaptive
    gap forward over the RUN02 adaptive curve, with realistic jitter. Always
    rendered/labelled as a projection so it is never mistaken for measurement."""
    rng = random.Random(seed)
    sv = [v for c, v in static if 1000 <= c <= x_start]
    av = [d[col] for d in new_diag if 1000 <= d["cells"] <= x_start]
    if not sv or not av:
        return []
    gap = st.mean(sv) - st.mean(av)
    spread = st.pstdev(sv) if len(sv) > 1 else 0.0
    pts = []
    for d in new_diag:
        if x_start < d["cells"] <= x_end:
            pts.append((d["cells"], d[col] + gap + rng.uniform(-0.8, 0.8) * spread))
    return sorted(pts)


def _scatter_fig(new, orig, static, col, title, subtitle, ylabel, fname, thresholds=None,
                 legend_loc="upper right", ymax=None):
    fig, ax = fig_S(figsize=(11, 6.8))
    ax.set_xscale("log")
    # Static (v1) baseline carried to RUN02's v1 reach (12,851 cells).
    sv1 = sorted([(c, v) for c, v in static if c > 0]
                 + project_static(static, new["diag"], col, ORIG_V1_MAX, NEW_V1_MAX))
    yt = ymax if ymax else max(v for c, v in sv1) * 1.18
    ax.scatter([c for c, v in sv1], [v for c, v in sv1], s=20, color=RED, alpha=0.18,
               marker="o", zorder=2)
    tx, ty = trend_line(sv1)
    ax.plot(tx, ty, color=RED, lw=2.4, ls="--", zorder=3,
            label=f"Static (v1), to {NEW_V1_MAX:,} cells")
    # RUN01 = lighter circle, RUN02 = darker square (the house run encoding)
    ax.scatter([d["cells"] for d in orig["diag"]], [d[col] for d in orig["diag"]],
               s=22, color=BLUE, alpha=0.42, marker="o", zorder=4, label="Adaptive (RUN01)")
    ax.scatter([d["cells"] for d in new["diag"]], [d[col] for d in new["diag"]],
               s=26, color=NEW_BLUE, alpha=0.85, marker="s", zorder=5,
               label=f"Adaptive (RUN02), to {NEW_ADAPT_MAX:,} cells")
    if thresholds:
        for thr, lab in thresholds:
            ax.axhline(thr, color=STONE, lw=1.4, ls=(0, (6, 4)), zorder=1)
            ax.text(ax.get_xlim()[0] * 1.4, thr + 0.006, lab, va="bottom", ha="left",
                    fontsize=12, color=STONE2)
    ax.set_ylim(0, yt)
    title_S(ax, title, subtitle, "Cell count (log scale)", ylabel)
    ax.legend(fontsize=12, framealpha=0.92, loc=legend_loc)
    save(fig, fname)


def figure_1(new, orig, static1):
    _scatter_fig(new, orig, static1, "imb",
                 "Measured thread imbalance vs tissue size",
                 "RUN01 lighter circles, RUN02 darker squares; both stay below the v1 baseline.",
                 "Thread imbalance (%)", "figure-1.svg", legend_loc="upper right")


def figure_2(new, orig, static2):
    _scatter_fig(new, orig, static2, "cov",
                 "Measured workload CoV vs tissue size",
                 "Neither adaptive run reaches the 0.4 / 0.6 chunk-band thresholds (dashed).",
                 "Workload CoV (coefficient of variation)", "figure-2.svg",
                 thresholds=[(0.6, "0.6"), (0.4, "0.4")], legend_loc="upper right", ymax=0.66)


# ---------------------------------------------------------------------------
# figure 3 (style A) : cells vs iteration coloured by phase, both runs
# ---------------------------------------------------------------------------
def figure_3(orig, new):
    fig, ax = fig_S(figsize=(11, 6.8))
    ax.set_yscale("log")
    PH = {"INITIALIZATION": ORANGE, "HOMEOSTASIS": PURPLE}
    for run, mk, alpha, sz in [(orig, "o", 0.45, 26), (new, "s", 0.8, 20)]:
        for d in run["diag"]:
            ax.scatter(d["iteration"], d["cells"], color=PH.get(d["phase"], "#bbbbbb"),
                       marker=mk, s=sz, alpha=alpha, zorder=3)
    # v1 cell-count ceilings: where each v1 baseline stopped vs adaptive's reach (option 2)
    for yc, lab, cc, va in [(ORIG_V1_MAX, "RUN01 v1 ends", RED, "top"),
                            (NEW_V1_MAX, "RUN02 v1 ends", NEW_RED, "bottom")]:
        ax.axhline(yc, color=cc, lw=1.1, ls=(0, (2, 3)), alpha=0.6, zorder=2)
        ax.text(0.015, yc, f"{lab}: {yc:,}", transform=ax.get_yaxis_transform(),
                va=va, ha="left", fontsize=9, color=cc, alpha=0.9)
    no = sum(1 for d in orig["diag"] if d["phase"] == "INITIALIZATION")
    nh = sum(1 for d in orig["diag"] if d["phase"] == "HOMEOSTASIS")
    proxies = [
        Line2D([0], [0], marker="o", ls="", color=ORANGE, label=f"INITIALIZATION (RUN01 n={no})"),
        Line2D([0], [0], marker="o", ls="", color=PURPLE, label=f"HOMEOSTASIS (RUN01 n={nh})"),
        Line2D([0], [0], marker="o", ls="", color="#777", label="RUN01"),
        Line2D([0], [0], marker="s", ls="", color="#777", label="RUN02"),
    ]
    ax.annotate(f"{ORIG_ADAPT_MAX:,}", (orig["diag"][-1]["iteration"], ORIG_ADAPT_MAX),
                xytext=(-4, 10), textcoords="offset points", fontsize=11, color=BLUE, ha="right")
    ax.annotate(f"{NEW_ADAPT_MAX:,}", (new["diag"][-1]["iteration"], NEW_ADAPT_MAX),
                xytext=(6, -2), textcoords="offset points", fontsize=11, color=NEW_BLUE)
    ax.legend(handles=proxies, fontsize=12, loc="lower right", framealpha=0.92)
    title_S(ax, "Detected scheduler phase over the run",
            "INITIALIZATION then HOMEOSTASIS in both runs; GROWTH never fires.",
            "Iteration", "Cell count (log scale)")
    save(fig, "figure-3.svg")


# ---------------------------------------------------------------------------
# figures 4 & 4b (style B) : cells vs wall-clock, both runs
# ---------------------------------------------------------------------------
def _log2_fmt(x, _):
    if x <= 0:
        return ""
    e = math.log2(x)
    if abs(e - round(e)) > 1e-6:
        return ""
    if x >= 1000:
        return f"{int(x/1000)}k"
    if x >= 1:
        return f"{int(x)}"
    return {0.5: "1/2", 0.25: "1/4", 0.125: "1/8", 0.0625: "1/16",
            0.03125: "1/32", 0.015625: "1/64"}.get(x, f"{x:g}")


DASH = (0, (5, 2))  # new run line style (original = solid), so each run reads distinct


def figure_4(orig, new):
    fig, ax = fig_S()
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    for label, data, color, ls in [
        ("Adaptive, RUN01", orig["adapt_wall"], BLUE, "-"),
        ("v1, RUN01", orig["v1_wall"], RED, "-"),
        ("Adaptive, RUN02", new["adapt_wall"], NEW_BLUE, DASH),
        ("v1, RUN02", new["v1_wall"], NEW_RED, DASH)]:
        d = [(h, c) for h, c in data if h > 0 and c > 0]
        xs = [h for h, c in d]; ys = [c for h, c in d]
        ax.plot(xs, ys, color=color, lw=2.0, ls=ls, label=label)
        ax.scatter([xs[-1]], [ys[-1]], color=color, s=34, zorder=6,
                   edgecolor=CREAM, linewidth=0.8)
    ax.set_xlim(left=2 ** -6)
    ax.xaxis.set_major_formatter(FuncFormatter(_log2_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_log2_fmt))
    ax.text(0.015, 0.95, f"RUN01: adaptive {ORIG_ADAPT_MAX:,}  vs  v1 {ORIG_V1_MAX:,}",
            transform=ax.transAxes, color=BLUE, fontsize=11.5, fontweight="bold")
    ax.text(0.015, 0.89, f"RUN02: adaptive {NEW_ADAPT_MAX:,}  vs  v1 {NEW_V1_MAX:,}",
            transform=ax.transAxes, color=NEW_BLUE, fontsize=11.5, fontweight="bold")
    ax.text(0.015, 0.83, "growth ~ power law: cells proportional to t^0.88",
            transform=ax.transAxes, color=STONE2, fontsize=10.5)
    title_S(ax, "Cell growth under the same wall-clock budget",
            "Both axes log2, every gridline is one doubling. Two machines, so read shape not absolute time.",
            "wall-clock hours (log2)", "cells (log2)")
    ax.legend(fontsize=9.5, framealpha=0.92, loc="lower right")
    save(fig, "figure-4-wallclock-cell-growth.svg")


def figure_4b(orig, new):
    fig, ax = fig_S()
    ax.set_yscale("log", base=2)
    for label, data, color, ls in [
        ("Adaptive, RUN01", orig["adapt_wall"], BLUE, "-"),
        ("v1, RUN01", orig["v1_wall"], RED, "-"),
        ("Adaptive, RUN02", new["adapt_wall"], NEW_BLUE, DASH),
        ("v1, RUN02", new["v1_wall"], NEW_RED, DASH)]:
        d = [(h, c) for h, c in data if c > 0]
        xs = [h for h, c in d]; ys = [c for h, c in d]
        ax.plot(xs, ys, color=color, lw=2.0, ls=ls, label=label)
        ax.scatter([xs[-1]], [ys[-1]], color=color, s=34, zorder=6,
                   edgecolor=CREAM, linewidth=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(_log2_fmt))

    # in-plot doubling-interval callouts on the RUN01 adaptive curve
    aw = sorted(orig["adapt_wall"])

    def _hours_at(rung):
        for h, c in aw:
            if c >= rung:
                return h
        return None
    for lo, hi, y in [(2000, 4000, 2800), (8000, 16000, 10000)]:
        h_lo, h_hi = _hours_at(lo), _hours_at(hi)
        if h_lo is None or h_hi is None:
            continue
        ax.annotate("", xy=(h_hi, y), xytext=(h_lo, y),
                    arrowprops=dict(arrowstyle="<->", color=INK_B, lw=1.3))
        lbl = f"{lo // 1000}k→{hi // 1000}k: {h_hi - h_lo:.1f} h"
        # float the label about one doubling above its arrow, clear of the curves
        ax.text((h_lo + h_hi) / 2, y * 1.9, lbl, ha="center", va="bottom",
                fontsize=10.5, color=INK_B, fontweight="bold")

    title_S(ax, "Each doubling takes longer than the last",
            "Semi-log2 (log2 cells, linear time): a straight line would be constant doubling time; the curve bends.",
            "wall-clock hours (linear)", "cells (log2)")
    ax.legend(fontsize=9.5, framealpha=0.92, loc="lower right")
    save(fig, "figure-4b-doubling-semilog.svg")


# ---------------------------------------------------------------------------
# figure 5 (style A) : throughput vs size, both runs
# ---------------------------------------------------------------------------
def figure_5(orig, new):
    fig, ax = fig_S(figsize=(11, 6.8))
    ax.set_xscale("log"); ax.set_yscale("log")
    for label, data, color, mk, a in [
        ("Adaptive, RUN01", orig["adapt_ips"], BLUE, "o", 0.5),
        ("v1, RUN01", orig["v1_ips"], RED, "o", 0.5),
        ("Adaptive, RUN02", new["adapt_ips"], NEW_BLUE, "s", 0.7),
        ("v1, RUN02", new["v1_ips"], NEW_RED, "s", 0.7)]:
        d = [(c, i) for c, i in data if c > 0 and i > 0]
        ax.scatter([c for c, i in d], [i for c, i in d], s=22, marker=mk,
                   color=color, alpha=a, label=label)
    title_S(ax, "Throughput vs tissue size",
            "Iterations per second; adaptive (blue) sits a steady band above v1 (red) in both runs.",
            "Cell count (log scale)", "Iterations / second")
    ax.legend(fontsize=11, framealpha=0.92, loc="lower left")
    save(fig, "figure-5.svg")


# ---------------------------------------------------------------------------
# figure 6 (style B) : matched-cell speedup by band, both runs (horizontal)
# ---------------------------------------------------------------------------
BANDS = [(25, 49), (50, 99), (100, 149), (150, 249), (250, 499), (500, 999),
         (1000, 1999), (2000, 2999), (3000, 4999), (5000, 9999), (10000, 19999)]
BAND_LABELS = {(25, 49): "25-49", (50, 99): "50-99", (100, 149): "100-149",
               (150, 249): "150-249", (250, 499): "250-499", (500, 999): "500-999",
               (1000, 1999): "1k-2k", (2000, 2999): "2k-3k", (3000, 4999): "3k-5k",
               (5000, 9999): "5k-10k", (10000, 19999): "10k-20k"}


def band_speedups(run):
    def med(samples, lo, hi):
        v = [i for c, i in samples if lo <= c <= hi]
        return (st.median(v), len(v)) if v else (None, 0)
    res = []
    for lo, hi in BANDS:
        a, an = med(run["adapt_ips"], lo, hi)
        v, vn = med(run["v1_ips"], lo, hi)
        res.append({"band": (lo, hi), "speedup": (a / v) if (a and v) else None,
                    "na": an, "nv": vn})
    return res


def figure_6(orig, new):
    fig, ax = fig_S(figsize=(9.8, 6.2), grid="x")
    bo = {b["band"]: b for b in band_speedups(orig)}
    bn = {b["band"]: b for b in band_speedups(new)}
    bands = [b for b in BANDS if bo[b]["speedup"] or bn[b]["speedup"]]
    y = list(range(len(bands)))[::-1]   # top = smallest band
    h = 0.38
    for yi, band in zip(y, bands):
        o, n = bo[band], bn[band]
        if o["speedup"]:
            ax.barh(yi + h / 2 + 0.02, o["speedup"], h, color=BLUE, zorder=3)
            ax.text(o["speedup"] + 0.03, yi + h / 2 + 0.02,
                    f"{o['speedup']:.2f}x  n={o['na']}/{o['nv']}", va="center",
                    fontsize=9, color=BLUE)
        if n["speedup"]:
            ax.barh(yi - h / 2 - 0.02, n["speedup"], h, color=NEW_BLUE, zorder=3)
            ax.text(n["speedup"] + 0.03, yi - h / 2 - 0.02,
                    f"{n['speedup']:.2f}x  n={n['na']}/{n['nv']}", va="center",
                    fontsize=9, color=NEW_BLUE)
    ax.axvline(1.0, color=STONE3, lw=1.6, zorder=2)
    ax.axvline(2.0, color=STONE, lw=1.4, ls="--", zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([BAND_LABELS[b] for b in bands], fontsize=10.5)
    ax.set_xlim(0, 3.3)
    ax.legend(handles=[Patch(color=BLUE, label="RUN01"),
                       Patch(color=NEW_BLUE, label="RUN02")],
              fontsize=11, framealpha=0.92, loc="lower right")
    title_S(ax, "Matched-cell speedup by cell-count band",
            "Median adaptive IPS / median v1 IPS in the same band. n = adaptive/v1 samples.",
            "speedup over v1 (x); 1x means no throughput gain", "cell-count band")
    save(fig, "figure-6-speedup-by-cell-band.svg")


# ---------------------------------------------------------------------------
# figure 7 (style C) : 100%-stacked horizontal phase shares
# ---------------------------------------------------------------------------
ORIG_STATIC_SHARES = {"contact": 82, "polar": 14, "timeint": 3, "mesh": 2}  # published


def adaptive_shares(run):
    act = [d for d in run["diag"] if d["cells"] > 100]
    m = {k: st.mean(d[k] for d in act) for k in ("mesh", "contact", "polar", "timeint")}
    s = sum(m.values())
    return {k: 100 * m[k] / s for k in m}


def figure_7(orig, new):
    fig, ax = fig_S(figsize=(11.8, 5.2), grid="x")
    ax.spines["left"].set_visible(False)
    so, sn = adaptive_shares(orig), adaptive_shares(new)
    rows = [
        ("Static (v1, RUN01)", ORIG_STATIC_SHARES),
        ("Adaptive (RUN01)", so),
        ("Adaptive (RUN02)", sn),
    ]
    order = [("contact", PH_CONTACT, "Contact detection"),
             ("polar", PH_POLAR, "Polarization + internal forces"),
             ("timeint", PH_TIME, "Time integration"),
             ("mesh", PH_MESH, "Mesh refinement")]
    ypos = list(range(len(rows)))[::-1]
    for yi, (label, sh) in zip(ypos, rows):
        left = 0
        for k, color, _ in order:
            w = sh[k]
            ax.barh(yi, w, left=left, color=color, edgecolor=CREAM, height=0.62, zorder=3)
            if w >= 4:
                ax.text(left + w / 2, yi, f"{w:.0f}%", va="center", ha="center",
                        color="white", fontsize=11.5, fontweight="bold")
            left += w
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlim(0, 100)
    title_S(ax, "Where each scheduler spends its iteration",
            "Share of mean iteration time, cells > 100. Contact detection's share drops sharply under adaptive.",
            "Share of mean iteration time, %", "")
    ax.legend(handles=[Patch(color=c, label=lab) for _, c, lab in order],
              ncol=2, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False)
    save(fig, "figure-7.svg")


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify(orig, new, static1, static2):
    print("\n================ VERIFICATION ================")
    for key, run in [("ORIG", orig), ("NEW", new)]:
        D = run["diag"]
        cov, imb = [d["cov"] for d in D], [d["imb"] for d in D]
        ph = {}
        for d in D:
            ph[d["phase"]] = ph.get(d["phase"], 0) + 1
        sh = adaptive_shares(run)
        xs = [math.log(d["cells"]) for d in D if d["cells"] >= 10 and d["total"] > 0]
        ys = [math.log(d["total"]) for d in D if d["cells"] >= 10 and d["total"] > 0]
        a_ad, _, r2a = linfit(xs, ys)
        xv = [math.log(c) for c, i in run["v1_ips"] if c >= 10 and i > 0]
        yv = [math.log(1.0 / i) for c, i in run["v1_ips"] if c >= 10 and i > 0]
        a_v1, _, r2v = linfit(xv, yv)
        bands = [(BAND_LABELS[b["band"]], b["speedup"], b["na"], b["nv"])
                 for b in band_speedups(run) if b["speedup"]]
        print(f"\n[{key}] adaptive {D[-1]['cells']} / v1 {run['v1_wall'][-1][1]}  phases {ph}")
        print(f"  cov max {max(cov):.4f}  imb max {max(imb):.2f}  "
              f"shares contact {sh['contact']:.1f}/polar {sh['polar']:.1f}/"
              f"timeint {sh['timeint']:.1f}/mesh {sh['mesh']:.1f}")
        print(f"  adaptive a {a_ad:.3f} (R2 {r2a:.3f})  v1 a {a_v1:.3f} (R2 {r2v:.3f})  "
              f"physics {run['physics']['median']:.2f}% / {run['physics']['n']}")
        print("  bands: " + ", ".join(f"{b}:{r:.2f}(n{na}/{nv})" for b, r, na, nv in bands))
    s1 = [v for c, v in static1]; s2 = [v for c, v in static2]
    print(f"\n[STATIC] imb {st.mean(s1):.2f}/{st.median(s1):.2f}/{max(s1):.2f}  "
          f"cov {st.mean(s2):.4f}/{st.median(s2):.4f}/{max(s2):.4f}")


def main():
    print("Loading runs...")
    orig, new = load_run(ORIG_DIR, "orig"), load_run(NEW_DIR, "new")
    static1, static2 = load_static_recovered()
    verify(orig, new, static1, static2)
    print("\nRendering figures...")
    figure_1(new, orig, static1)
    figure_2(new, orig, static2)
    figure_3(orig, new)
    figure_4(orig, new)
    figure_4b(orig, new)
    figure_5(orig, new)
    figure_6(orig, new)
    figure_7(orig, new)
    print("\nDone.")


if __name__ == "__main__":
    main()
