#!/usr/bin/env python3
"""
Render the bench2 figures for the SimuCell3D post from ./datapoints/*.csv
only (no run directories, no artifact access). The default run writes the four
figures the post embeds; add --memory for the fifth (resident memory). House
style follows the older, archived make_figures.py: cream background, bold left-aligned
title + grey subtitle, stone chrome, red = v1.0 family / stopped early,
blue = 2.x.

Every number printed on a figure is read from a CSV and rounded only when it is
printed ("round everywhere": about 130 cells, about 25k, 2.5 h, 16 to 17 h,
about N^1.1). Data positions use the exact CSV values. Build names are mapped
to the post's names at print time too, so the CSVs keep the artifact's wording.
Output is deterministic (fixed SVG hash salt, no date metadata), so running
twice gives identical files.

  python3 make_bench2_figures.py            # needs matplotlib (3.10.x used)
  python3 make_bench2_figures.py --memory   # also bench2-memory
Override the output dir with SIMUCELL3D_IMG_OUT.
"""
import csv
import math
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.join(HERE, "datapoints")
IMG_OUT = os.environ.get(
    "SIMUCELL3D_IMG_OUT",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "images", "blog", "simucell3d")),
)

plt.rcParams["svg.hashsalt"] = "simucell3d-bench2"

# palette: identical to ../make_figures.py
BLUE = "#4c72b0"
NEW_BLUE = "#2f4b7c"
RED = "#c44e52"
CREAM, INK_B = "#fbfaf7", "#1f2933"  # CREAM: page background, decorative (house token, kept)
# STONE (2.52:1 on white) is spine/cap-line chrome and the deliberately faint
# context series (dev builds), decorative by design; GRID_B is gridlines, decorative.
# Both are the house tokens from ../make_figures.py and must match the older figures.
STONE, STONE2, STONE3, GRID_B = "#a8a29e", "#6b7280", "#4b5563", "#e7e5df"  # decorative

# series keys as they appear in the CSVs
V1F, DEV, V21 = "v1.0 + crash fixes", "2.x dev builds", "v2.1.0"
# what the post calls them (printed labels only)
DISPLAY = {"v1.0": "v1.0 as tagged", V1F: "v1.0 + two patches"}
# how a run ended, in the post's neutral wording
OUTCOME = {"abort": "stopped early", "crash": "stopped early", "48 h cap": "two-day cap"}


def disp(series):
    return DISPLAY.get(series, series)


# ---------------------------------------------------------------------------
# io + helpers
# ---------------------------------------------------------------------------
def read_csv(name):
    with open(os.path.join(DP, name), newline="") as f:
        return list(csv.DictReader(f))


SETUP = {r["key"]: r["value"] for r in read_csv("bench2_setup.csv")}
# The benchmark time step is printed relative to the paper's (1e-7 s), as in the post.
PAPER_DT = 1e-7
assert abs(float(SETUP["dt"]) * 4 - PAPER_DT) < 1e-12, "dt is no longer a quarter of the paper's"
assert SETUP["case"].startswith("T1 ") and SETUP["stop_cells"] == "150k"
assert SETUP["time_cap_hours"] == "48"
CASE = SETUP["case"][3:].capitalize()          # drop the study's internal case id
SETUP_LINE = (f"{CASE} at a quarter of the paper's time step, {SETUP['threads']} threads on "
              f"{SETUP['machine']}.")
STOP_LINE = "runs stop at 150,000 cells or two days"


def about_cells(n, about=True):
    """Rounded cell count as it is said: nearest ten below a thousand, nearest thousand above."""
    n = float(n)
    txt = f"{round(n / 1000):d}k" if n >= 1000 else f"{round(n / 10) * 10:d}"
    return f"about {txt}" if about else txt


def cells_label(n):
    """'about 130 cells' below a thousand, 'about 25k' above (the k already reads as a count)."""
    return about_cells(n) + ("" if float(n) >= 1000 else " cells")


def hours_2sf(lo, hi=None):
    """One value to two significant figures; a reported range in whole hours."""
    lo = float(lo)
    if hi is not None and float(hi) > lo:
        return f"{round(lo):d} to {round(float(hi)):d} h"
    return f"{float(f'{lo:.2g}'):g} h"


def kfmt(x, _=None):
    """10 -> '10', 1000 -> '1k', 100000 -> '100k' (log-axis tick labels)."""
    if x >= 1000:
        return f"{x / 1000:g}k"
    return f"{x:g}"


def fig_S(figsize=(9.8, 5.6), grid="both"):
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
    ax.set_title(t, color=INK_B, fontsize=17, fontweight="bold", loc="left", pad=24)
    if sub:
        # kept out of tight_layout so a longer subtitle cannot resize the axes
        ax.text(0.0, 1.035, sub, transform=ax.transAxes, color=STONE2,
                fontsize=11.5).set_in_layout(False)
    ax.set_xlabel(x, color=STONE2, fontsize=12.5)
    ax.set_ylabel(y, color=STONE2, fontsize=12.5)


def save(fig, name):
    os.makedirs(IMG_OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_OUT, name), format="svg", facecolor=fig.get_facecolor(),
                metadata={"Date": None})
    png = name[:-4] + ".png"
    fig.savefig(os.path.join(IMG_OUT, png), format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {name} + {png}")


def runs_of(rows, xkey, ykey):
    """Group 'curve' rows into OrderedDict[(series, build, run)] -> [(x, y)]."""
    out = OrderedDict()
    for r in rows:
        if r["kind"] != "curve":
            continue
        out.setdefault((r["series"], r["build"], r["run"]), []).append(
            (float(r[xkey]), float(r[ykey])))
    return out


def style_for(series):
    """(color, lw, alpha, zorder) per series: dev builds faint and behind."""
    return {V1F: (RED, 1.8, 0.95, 4), V21: (NEW_BLUE, 1.8, 0.95, 5),
            DEV: (STONE, 1.1, 0.55, 2)}[series]


# ---------------------------------------------------------------------------
# 1. reach: final cells per build
# ---------------------------------------------------------------------------
def figure_reach():
    rows = read_csv("bench2_reach.csv")
    fig, ax = fig_S(figsize=(10.4, 6.6), grid="x")
    ax.set_xscale("log")
    left = 10
    # one row per run; a gap between build groups
    ypos, y, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r["series"] != prev:
            y += 0.55
        ypos.append(y)
        y += 1.0
        prev = r["series"]
    n_in = {}
    for r in rows:
        n_in[r["series"]] = n_in.get(r["series"], 0) + 1
    labels = []
    dev_i = 0
    for r in rows:
        s = r["series"]
        if s == DEV:
            dev_i += 1
            labels.append(f"2.x dev build {dev_i}")
        elif n_in[s] > 1:
            labels.append(f"{disp(s)}, run {r['run']}")
        else:
            labels.append(disp(s))
    for yi, r in zip(ypos, rows):
        failed = r["outcome"] in ("abort", "crash")
        color = RED if failed else (NEW_BLUE if r["series"] == V21 else BLUE)
        val = float(r["final_cells"])
        ax.barh(yi, val - left, 0.62, left=left, color=color, zorder=3)
        ax.text(val * 1.12, yi, f"{cells_label(val)}, {OUTCOME[r['outcome']]}",
                va="center", ha="left", fontsize=10.5, color=color,
                fontweight="bold" if failed else "normal")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(left, 3.0e6)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1e3, 1e4, 1e5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    lo, hi = ypos[-1] + 0.7, ypos[0] - 0.7
    ax.set_ylim(lo, hi)
    # legend in the free space right of the short v2.0 bar
    y_v20 = ypos[[r["series"] for r in rows].index("v2.0")]
    ax.legend(handles=[Patch(color=RED, label="stopped early"),
                       Patch(color=BLUE, label="2.x dev build, ran to the two-day cap"),
                       Patch(color=NEW_BLUE, label="v2.1.0, ran to the two-day cap")],
              fontsize=10.5, framealpha=0.92, loc="center right",
              bbox_to_anchor=(1.0, (lo - y_v20) / (lo - hi)), bbox_transform=ax.transAxes)
    title_S(ax, "How far each build grows the benchmark tissue",
            SETUP_LINE,
            "final cell count (log scale); " + STOP_LINE, "")
    save(fig, "bench2-reach.svg")


# ---------------------------------------------------------------------------
# 2. growth: cells vs wall-clock hours
# ---------------------------------------------------------------------------
def figure_growth():
    rows = read_csv("bench2_growth.csv")
    fig, ax = fig_S()
    ax.set_yscale("log")
    curves = runs_of(rows, "hours", "cells")
    for series in (DEV, V1F, V21):
        c, lw, a, z = style_for(series)
        for (s, b, r), pts in curves.items():
            if s != series:
                continue
            pts = [(h, n) for h, n in pts if n >= 100]
            ax.plot([h for h, n in pts], [n for h, n in pts], color=c, lw=lw, alpha=a, zorder=z)
    cap = next(float(r["hours"]) for r in rows if r["kind"] == "time_cap")
    ax.axvline(cap, color=STONE, lw=1.4, ls=(0, (6, 4)), zorder=1)
    ax.text(cap - 0.6, 160, f"two-day cap ({cap:g} h)", rotation=90, ha="right", va="bottom",
            fontsize=10.5, color=STONE2)
    aborts = [r for r in rows if r["kind"] == "abort_marker"]
    ax.scatter([float(r["hours"]) for r in aborts], [float(r["cells"]) for r in aborts],
               marker="X", s=90, color=RED, edgecolor=CREAM, linewidth=0.9, zorder=7)
    # the two early stops sit 6 px apart, so they share one label
    early = sorted((r for r in aborts if float(r["hours"]) < 12), key=lambda r: float(r["cells"]))
    late = [r for r in aborts if float(r["hours"]) >= 12]
    e_h = max(float(r["hours"]) for r in early)
    e_n = min(float(r["cells"]) for r in early)
    ax.text(e_h + 0.8, e_n * 0.80,
            "stopped at about "
            + " and ".join(about_cells(r["cells"], about=False) for r in early) + " cells",
            ha="left", va="top", fontsize=10.5, color=RED, fontweight="bold")
    for r in late:
        ax.text(float(r["hours"]) + 0.8, float(r["cells"]) * 0.80,
                f"stopped at {about_cells(r['cells'])} cells",
                ha="left", va="top", fontsize=10.5, color=RED, fontweight="bold")
    nrun = {s: len({k for k in curves if k[0] == s}) for s in (V1F, DEV, V21)}
    # series key in the empty lower-right corner (fast early growth fills the upper left)
    ax.text(0.90, 0.30, f"{V21}, {nrun[V21]} runs: ran to the two-day cap", ha="right",
            transform=ax.transAxes, color=NEW_BLUE, fontsize=11.5, fontweight="bold")
    ax.text(0.90, 0.24, f"{disp(V1F)}, {nrun[V1F]} runs: all stopped early", ha="right",
            transform=ax.transAxes, color=RED, fontsize=11.5, fontweight="bold")
    ax.text(0.90, 0.18, f"{DEV}, {nrun[DEV]} runs (faint)", ha="right",
            transform=ax.transAxes, color=STONE2, fontsize=10.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(100, 2.0e5)
    ax.yaxis.set_major_locator(FixedLocator([100, 1e3, 1e4, 1e5]))
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    title_S(ax, "Cell growth against wall-clock time",
            SETUP_LINE,
            "wall-clock hours (curves are downsampled)", "cells (log scale)")
    save(fig, "bench2-growth.svg")


# ---------------------------------------------------------------------------
# 3. seconds per iteration vs cells
# ---------------------------------------------------------------------------
def figure_sec_per_iter():
    rows = read_csv("bench2_sec_per_iter.csv")
    fig, ax = fig_S()
    ax.set_xscale("log"); ax.set_yscale("log")
    curves = runs_of(rows, "cells", "sec_per_iter")
    for series in (DEV, V1F, V21):
        c, lw, a, z = style_for(series)
        for (s, b, r), pts in curves.items():
            if s == series:
                ax.plot([x for x, y in pts], [y for x, y in pts], color=c, lw=lw, alpha=a, zorder=z)
    # reference slopes: the artifact's stated exponent range, drawn through a
    # point half a decade under v2.1.0 run 1 at ~10k cells so they sit clear of the data
    v21r1 = curves[(V21, "B7", "1")]
    ax0, ay0 = min(v21r1, key=lambda p: abs(math.log10(p[0] / 1e4)))
    ay0 /= 10 ** 0.5
    xs = [2.0e3, 1.6e5]
    for k in (float(r["exponent"]) for r in rows if r["kind"] == "reference_exponent"):
        ax.plot(xs, [ay0 * (x / ax0) ** k for x in xs], color=STONE3, lw=0.9, ls=(0, (2, 2)), zorder=3)
    ks = [f"{float(r['exponent']):.1f}" for r in rows if r["kind"] == "reference_exponent"]
    ax.text(0.98, 0.05, f"dotted: reference slopes, about $N^{{{ks[0]}}}$ and $N^{{{ks[1]}}}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=STONE3)
    nrun = {s: len({k for k in curves if k[0] == s}) for s in (V1F, DEV, V21)}
    ax.text(0.015, 0.95, f"{disp(V1F)}, {nrun[V1F]} runs", transform=ax.transAxes,
            color=RED, fontsize=11.5, fontweight="bold")
    ax.text(0.015, 0.89, f"{V21}, {nrun[V21]} runs", transform=ax.transAxes,
            color=NEW_BLUE, fontsize=11.5, fontweight="bold")
    ax.text(0.015, 0.83, f"{DEV}, {nrun[DEV]} runs (faint)", transform=ax.transAxes,
            color=STONE2, fontsize=10.5)
    ax.set_xlim(1.0e3, 2.0e5)
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    title_S(ax, "Cost of one iteration as the tissue grows",
            SETUP_LINE + " Both axes log.",
            "cells (log scale)", "seconds per iteration (log scale)")
    save(fig, "bench2-sec-per-iter.svg")


# ---------------------------------------------------------------------------
# 4. matched cases: dumbbell of hours
# ---------------------------------------------------------------------------
def figure_matched():
    rows = read_csv("bench2_matched_cases.csv")
    fig, ax = fig_S(figsize=(11.6, 5.8), grid="x")
    ax.spines["left"].set_visible(False)
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        v_lo, v_hi = float(r["v1_fix_hours_lo"]), float(r["v1_fix_hours_hi"])
        n_lo, n_hi = float(r["v210_hours_lo"]), float(r["v210_hours_hi"])
        ax.plot([n_hi, v_lo], [yi, yi], color=STONE, lw=1.6, zorder=1, solid_capstyle="round")
        for lo, hi, c in ((v_lo, v_hi, RED), (n_lo, n_hi, NEW_BLUE)):
            if hi > lo:   # a reported spread: draw it as a capsule
                ax.plot([lo, hi], [yi, yi], color=c, lw=9, solid_capstyle="round", zorder=4)
            else:
                ax.scatter([lo], [yi], s=120, color=c, zorder=5)
        v_txt = hours_2sf(v_lo, v_hi)
        n_txt = hours_2sf(n_lo, n_hi)
        ax.text(v_hi + 0.45, yi, v_txt, va="center", ha="left", fontsize=11, color=RED)
        ax.text(n_lo - 0.45, yi, n_txt, va="center", ha="right", fontsize=11, color=NEW_BLUE,
                fontweight="bold")
        ax.text(27.6, yi, f"{r['ratio_verbatim']}x", va="center", ha="right", fontsize=12,
                fontweight="bold", color=INK_B)
    ax.text(27.6, y[0] + 0.62, "ratio", va="center", ha="right", fontsize=10.5, color=STONE2)
    labels = [f"{r['case'].replace('eps ', 'ε ')}, Arm {r['arm']}" for r in rows]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlim(-3.2, 28)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=RED, markersize=10, label=disp(V1F)),
                       Line2D([0], [0], marker="o", ls="", color=NEW_BLUE, markersize=10, label=V21),
                       Line2D([0], [0], color=STONE3, lw=7, solid_capstyle="round",
                              label="bar: reported range")],
              fontsize=10.5, framealpha=0.92, loc="upper center", bbox_to_anchor=(0.6, 1.0), ncol=3)
    title_S(ax, "The same cases, run on both versions",
            f"Hours, {SETUP['threads']} threads on {SETUP['machine']}. "
            "Arm A: paper parameters; Arm B: literature parameters.",
            "wall-clock hours", "")
    save(fig, "bench2-matched-cases.svg")


# ---------------------------------------------------------------------------
# 5. memory vs cells
# ---------------------------------------------------------------------------
def figure_memory():
    rows = read_csv("bench2_memory.csv")
    S = [r for r in rows if r["kind"] == "sample"]
    cap = next(float(r["rss_gib"]) for r in rows if r["kind"] == "memory_cap")
    slope = next(r["rss_gib"] for r in rows if r["kind"] == "v210_slope_gib_per_1k_cells")
    fig, ax = fig_S()
    groups = [
        ("early 2.x dev builds (3 builds)", ("B3", "B4", "B5"), BLUE, "o", 34),
        ("last 2.x dev build", ("B6",), STONE3, "D", 30),
        (V21, ("B7",), NEW_BLUE, "s", 34),
        (disp(V1F), ("B1",), RED, "o", 34),
    ]
    for lab, builds, c, mk, sz in groups:
        pts = [(float(r["cells"]), float(r["rss_gib"])) for r in S if r["build"] in builds]
        ax.scatter([x for x, _ in pts], [g for _, g in pts], marker=mk, s=sz, color=c,
                   edgecolor=CREAM, linewidth=0.6, zorder=4, label=lab)
    ax.axhline(cap, color=STONE, lw=1.4, ls=(0, (6, 4)), zorder=1)
    ax.text(2000, cap + 0.8, f"{cap:g} GiB memory cap", ha="left", va="bottom",
            fontsize=10.5, color=STONE2)
    xs = [0, 160000]
    ax.plot(xs, [float(slope) * x / 1000 for x in xs], color=NEW_BLUE, lw=1.0,
            ls=(0, (2, 2)), zorder=2)
    ax.text(150000, float(slope) * 150 + 3.2, f"{slope} GiB per 1,000 cells", ha="right",
            va="bottom", fontsize=10.5, color=NEW_BLUE)
    ax.set_xlim(0, 160000)
    ax.set_ylim(0, 66)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: kfmt(x) if x else "0"))
    ax.legend(fontsize=10.5, framealpha=0.92, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    title_S(ax, "Resident memory as the tissue grows",
            SETUP_LINE + " One point per memory sample.",
            "cells", "resident memory (GiB)")
    save(fig, "bench2-memory.svg")


def main():
    print("Rendering bench2 figures...")
    figure_reach()
    figure_growth()
    figure_sec_per_iter()
    figure_matched()
    if "--memory" in sys.argv[1:]:   # not embedded in the post, so not written by default
        figure_memory()
    print("Done.")


if __name__ == "__main__":
    main()
