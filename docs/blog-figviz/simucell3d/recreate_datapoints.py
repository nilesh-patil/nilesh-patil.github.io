#!/usr/bin/env python3
"""
Recreate blog-figure datapoints from the NEW SimuCell3D run
parallel_benchmark_20260621_055502 (7 cores @3.10GHz, paper_exact_fast_growth).

Stdlib only (csv + statistics + math). No pandas/numpy dependency so this is
fully reproducible on a bare interpreter. Least-squares fits are implemented
inline.

Outputs:
  datapoints/fig1_imbalance.csv
  datapoints/fig2_cov.csv
  datapoints/fig3_phase.csv
  datapoints/fig4_wallclock.csv
  datapoints/fig4b_doubling.csv
  datapoints/fig5_ips.csv
  datapoints/fig6_speedup_bands.csv
  datapoints/fig7_phase_shares.csv
  datapoints/physics_check.csv
  03_new_run_recreated_summary.json
"""
import csv
import json
import math
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get(
    "SIMUCELL3D_RESULTS",
    "/Users/nilesh-patil/versioned-projects/Github-personal.tmp/simucell3d/docs/simulation_results",
)
RUN = os.path.join(RESULTS, "parallel_benchmark_20260621_055502")
OUT = HERE
DP = os.path.join(OUT, "datapoints")
os.makedirs(DP, exist_ok=True)

DIAG = os.path.join(RUN, "sim_adaptive", "performance_diagnostics.csv")
V1LOG = os.path.join(RUN, "logs", "v1.log")
V1STATS = os.path.join(RUN, "sim_v1", "simulation_statistics.csv")
ADBIO = os.path.join(RUN, "plots", "updated", "cleaned_data", "adaptive_biological_cleaned.csv")
V1BIO = os.path.join(RUN, "plots", "updated", "cleaned_data", "v1_biological_cleaned.csv")

# concurrent / fair endpoints (lead ground truth)
ADAPTIVE_FINAL_CELLS = 26534      # iter 21900
V1_CONCURRENT_CELLS = 12851       # iter 20100 (excludes unfair _solo_tail)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def linfit(xs, ys):
    """Ordinary least squares y = a + b*x. Returns (b_slope, a_intercept, r2)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return b, a, r2


def parse_hms(s):
    """HH:MM:SS where HH may exceed 24 -> seconds (float)."""
    s = s.strip()
    parts = s.split(":")
    h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + sec


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def crosses_at(pairs, threshold):
    """First (x, y) where y >= threshold, given pairs sorted by x. pairs=(x,y)."""
    for x, y in pairs:
        if y >= threshold:
            return x, y
    return None, None


# ----------------------------------------------------------------------------
# load adaptive diagnostics (by column NAME - schema has sim_time/wall_epoch lead)
# ----------------------------------------------------------------------------
adiag = []
with open(DIAG) as f:
    r = csv.DictReader(f)
    for row in r:
        adiag.append({
            "iteration": int(row["iteration"]),
            "cells": int(row["cells"]),
            "cov": float(row["cov"]),
            "phase": row["phase"].strip(),
            "wall_epoch": int(row["wall_epoch"]),
            "mesh": float(row["mesh_refinement_ms"]),
            "contact": float(row["contact_detection_ms"]),
            "polar": float(row["polarization_internal_forces_ms"]),
            "timeint": float(row["time_integration_ms"]),
            "total": float(row["total_iteration_ms"]),
            "imb": float(row["thread_imbalance_pct"]),
        })
adiag.sort(key=lambda d: d["iteration"])
wall0 = adiag[0]["wall_epoch"]

summary = {"run": os.path.basename(RUN), "n_adaptive_diag_rows": len(adiag)}

# ----------------------------------------------------------------------------
# v1.log : iteration -> cells
# ----------------------------------------------------------------------------
v1_iter_cells = {}
pat = re.compile(r"iteration:\s*(\d+),.*nb cells\s*(\d+)")
with open(V1LOG, errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            v1_iter_cells[int(m.group(1))] = int(m.group(2))

# ----------------------------------------------------------------------------
# v1 simulation_statistics : distinct iteration -> cumulative seconds
# ----------------------------------------------------------------------------
v1_iter_secs = {}
v1_iter_pressure = {}
with open(V1STATS) as f:
    r = csv.DictReader(f)
    # accumulate per-iteration: time is identical across cells of same iter
    pres_acc = {}
    for row in r:
        it = int(row["iteration"])
        if it not in v1_iter_secs:
            v1_iter_secs[it] = parse_hms(row["computation_time_(hh::mm:ss)"])
        try:
            p = float(row["pressure"])
            pres_acc.setdefault(it, []).append(p)
        except (ValueError, KeyError):
            pass
    for it, ps in pres_acc.items():
        v1_iter_pressure[it] = sum(ps) / len(ps)

# ============================================================================
# FIGURE 1 - adaptive thread_imbalance_pct vs cells (v1 not available)
# ============================================================================
rows = [(d["iteration"], d["cells"], d["imb"]) for d in adiag]
write_csv(os.path.join(DP, "fig1_imbalance.csv"),
          ["iteration", "cells", "thread_imbalance_pct"], rows)

imb_all = [d["imb"] for d in adiag]
imb_active = [d["imb"] for d in adiag if d["cells"] > 100]  # meaningful regime
summary["fig1_imbalance"] = {
    "v1_available": False,
    "v1_note": "genuine v1.0 emits no performance_diagnostics.csv / thread_imbalance_pct anywhere in the new run",
    "adaptive_all_rows": {
        "n": len(imb_all),
        "mean": round(st.mean(imb_all), 4),
        "median": round(st.median(imb_all), 4),
        "max": round(max(imb_all), 4),
    },
    "adaptive_cells_gt_100": {
        "n": len(imb_active),
        "mean": round(st.mean(imb_active), 4),
        "median": round(st.median(imb_active), 4),
        "max": round(max(imb_active), 4),
    },
}

# ============================================================================
# FIGURE 2 - adaptive cov vs cells (v1 not available); thresholds 0.4/0.6
# ============================================================================
rows = [(d["iteration"], d["cells"], d["cov"]) for d in adiag]
write_csv(os.path.join(DP, "fig2_cov.csv"),
          ["iteration", "cells", "cov"], rows)

cov_all = [d["cov"] for d in adiag]
cov_active = [d["cov"] for d in adiag if d["cells"] > 100]
summary["fig2_cov"] = {
    "v1_available": False,
    "v1_note": "no cov column exists for v1 in the new run",
    "adaptive_all_rows": {
        "n": len(cov_all),
        "mean": round(st.mean(cov_all), 4),
        "median": round(st.median(cov_all), 4),
        "max": round(max(cov_all), 4),
    },
    "adaptive_cells_gt_100": {
        "n": len(cov_active),
        "mean": round(st.mean(cov_active), 4),
        "median": round(st.median(cov_active), 4),
        "max": round(max(cov_active), 4),
    },
    "ever_ge_0.4": max(cov_all) >= 0.4,
    "ever_ge_0.6": max(cov_all) >= 0.6,
    "thresholds_drawn": [0.4, 0.6],
}

# ============================================================================
# FIGURE 3 - scheduler phase over the run
# ============================================================================
rows = [(d["iteration"], d["cells"], d["phase"]) for d in adiag]
write_csv(os.path.join(DP, "fig3_phase.csv"),
          ["iteration", "cells", "phase"], rows)

phase_counts = {}
for d in adiag:
    phase_counts[d["phase"]] = phase_counts.get(d["phase"], 0) + 1

# transition: first row whose phase != first phase
first_phase = adiag[0]["phase"]
trans = None
for d in adiag:
    if d["phase"] != first_phase:
        trans = d
        break
last_init = max((d for d in adiag if d["phase"] == "INITIALIZATION"),
                key=lambda d: d["iteration"])
summary["fig3_phase"] = {
    "phase_counts": phase_counts,
    "growth_ever": "GROWTH" in phase_counts,
    "last_INITIALIZATION_iter": last_init["iteration"],
    "last_INITIALIZATION_cells": last_init["cells"],
    "first_HOMEOSTASIS_iter": trans["iteration"] if trans else None,
    "first_HOMEOSTASIS_cells": trans["cells"] if trans else None,
    "final_iter": adiag[-1]["iteration"],
    "final_cells": adiag[-1]["cells"],
}

# ============================================================================
# FIGURE 4 / 4b - cells vs wall-clock for BOTH modes
# ============================================================================
# adaptive: wall hours from wall_epoch
adapt_wc = [(round((d["wall_epoch"] - wall0) / 3600.0, 5), d["cells"], d["iteration"])
            for d in adiag]
# v1: join distinct stats iterations (cumulative secs) with v1.log cells
v1_wc = []
for it in sorted(v1_iter_secs):
    if it in v1_iter_cells:
        hrs = v1_iter_secs[it] / 3600.0
        v1_wc.append((round(hrs, 5), v1_iter_cells[it], it))

rows = []
for h, c, it in adapt_wc:
    rows.append(["adaptive", it, c, h])
for h, c, it in v1_wc:
    rows.append(["v1", it, c, h])
write_csv(os.path.join(DP, "fig4_wallclock.csv"),
          ["mode", "iteration", "cells", "wall_hours"], rows)

# power-law fit adaptive: cells ~ t^k  (log-log, t>0 & cells>0)
xs = [math.log(h) for h, c, it in adapt_wc if h > 0 and c > 0]
ys = [math.log(c) for h, c, it in adapt_wc if h > 0 and c > 0]
k_ad, _, r2_ad = linfit(xs, ys)

summary["fig4_wallclock"] = {
    "adaptive_final": {"iter": adapt_wc[-1][2], "cells": adapt_wc[-1][1],
                       "wall_hours": adapt_wc[-1][0]},
    "v1_concurrent_final": {"iter": v1_wc[-1][2], "cells": v1_wc[-1][1],
                            "wall_hours": v1_wc[-1][0]},
    "adaptive_powerlaw_cells_vs_t": {"exponent_k": round(k_ad, 4),
                                     "r2": round(r2_ad, 5)},
}

# 4b doubling: rungs on the adaptive (cells, hours) curve
adapt_sorted = sorted([(c, h) for h, c, it in adapt_wc])  # by cells
# helper: first wall-hour at which cells >= rung
def hours_at_cells(rung):
    for h, c, it in adapt_wc:
        if c >= rung:
            return h
    return None

rungs = [1000, 2000, 4000, 8000, 16000]
rung_hours = {r: hours_at_cells(r) for r in rungs}
doublings = []
for a, b in zip(rungs[:-1], rungs[1:]):
    ha, hb = rung_hours[a], rung_hours[b]
    if ha is not None and hb is not None:
        doublings.append({"from": a, "to": b, "delta_hours": round(hb - ha, 3),
                          "h_from": round(ha, 3), "h_to": round(hb, 3)})

# v1 rungs (caps ~12851)
def v1_hours_at_cells(rung):
    for h, c, it in v1_wc:
        if c >= rung:
            return h
    return None
v1_doublings = []
for a, b in zip(rungs[:-1], rungs[1:]):
    ha, hb = v1_hours_at_cells(a), v1_hours_at_cells(b)
    if ha is not None and hb is not None:
        v1_doublings.append({"from": a, "to": b, "delta_hours": round(hb - ha, 3)})

rows = []
for d in doublings:
    rows.append(["adaptive", d["from"], d["to"], d["h_from"], d["h_to"], d["delta_hours"]])
for d in v1_doublings:
    rows.append(["v1", d["from"], d["to"], "", "", d["delta_hours"]])
write_csv(os.path.join(DP, "fig4b_doubling.csv"),
          ["mode", "rung_from", "rung_to", "hours_from", "hours_to", "delta_hours"], rows)

# ratio of successive doubling intervals (adaptive)
ratios = []
for i in range(1, len(doublings)):
    prev = doublings[i - 1]["delta_hours"]
    cur = doublings[i]["delta_hours"]
    if prev > 0:
        ratios.append(round(cur / prev, 3))
summary["fig4b_doubling"] = {
    "adaptive_doublings": doublings,
    "adaptive_successive_ratios": ratios,
    "v1_doublings": v1_doublings,
}

# ============================================================================
# FIGURE 5 - IPS vs cells, both modes
# ============================================================================
# adaptive IPS = 1000 / total_iteration_ms
adapt_ips = [(d["cells"], 1000.0 / d["total"], d["iteration"]) for d in adiag if d["total"] > 0]
# cross-check: 100 iters / delta wall_epoch between consecutive sampled rows
cross = []
for i in range(1, len(adiag)):
    d0, d1 = adiag[i - 1], adiag[i]
    diter = d1["iteration"] - d0["iteration"]
    dt = d1["wall_epoch"] - d0["wall_epoch"]
    if dt > 0:
        cross.append((d1["cells"], diter / dt, 1000.0 / d1["total"]))
# mean ratio of (1000/total) vs (diter/dt)
ratio_cross = st.median([a / b for c, b, a in cross if b > 0]) if cross else float("nan")

# v1 IPS from computation_time deltas across distinct iterations, cells via v1.log
v1_iters_sorted = sorted(v1_iter_secs)
v1_ips = []
for i in range(1, len(v1_iters_sorted)):
    it0, it1 = v1_iters_sorted[i - 1], v1_iters_sorted[i]
    dt = v1_iter_secs[it1] - v1_iter_secs[it0]
    diter = it1 - it0
    if dt > 0 and it1 in v1_iter_cells:
        v1_ips.append((v1_iter_cells[it1], diter / dt, it1))

rows = []
for c, ips, it in adapt_ips:
    rows.append(["adaptive", it, c, round(ips, 6)])
for c, ips, it in v1_ips:
    rows.append(["v1", it, c, round(ips, 6)])
write_csv(os.path.join(DP, "fig5_ips.csv"),
          ["mode", "iteration", "cells", "iter_per_sec"], rows)

# scaling: time/iter ~ N^alpha. adaptive time/iter = total_iteration_ms
xs = [math.log(d["cells"]) for d in adiag if d["cells"] > 0 and d["total"] > 0]
ys = [math.log(d["total"]) for d in adiag if d["cells"] > 0 and d["total"] > 0]
alpha_ad, _, r2a = linfit(xs, ys)
# v1 time/iter = 1/IPS (seconds) -> convert nothing, fit log(sec_per_iter) vs log(cells)
xv = [math.log(c) for c, ips, it in v1_ips if c > 0 and ips > 0]
yv = [math.log(1.0 / ips) for c, ips, it in v1_ips if c > 0 and ips > 0]
alpha_v1, _, r2v = linfit(xv, yv)

summary["fig5_ips"] = {
    "adaptive_final_ips": round(adapt_ips[-1][1], 6),
    "adaptive_final_cells": adapt_ips[-1][0],
    "v1_final_ips": round(v1_ips[-1][1], 6),
    "v1_final_cells": v1_ips[-1][0],
    "adaptive_scaling_alpha_time_per_iter_vs_N": round(alpha_ad, 4),
    "adaptive_scaling_r2": round(r2a, 5),
    "v1_scaling_alpha_time_per_iter_vs_N": round(alpha_v1, 4),
    "v1_scaling_r2": round(r2v, 5),
    "crosscheck_adaptive_ips_total_vs_wallepoch_median_ratio": round(ratio_cross, 4),
    "n_adaptive_ips": len(adapt_ips),
    "n_v1_ips": len(v1_ips),
}

# ============================================================================
# FIGURE 6 - matched-cell speedup by band
# ============================================================================
BANDS = [(10, 24), (25, 49), (50, 99), (100, 249), (250, 499), (500, 999),
         (1000, 1999), (2000, 3999), (4000, 7999), (8000, 15999), (16000, 31999)]


def band_label(lo, hi):
    return f"{lo}-{hi}"


def median_in_band(samples, lo, hi):
    vals = [ips for c, ips, *_ in samples if lo <= c <= hi]
    return (st.median(vals), len(vals)) if vals else (None, 0)


rows = []
band_summary = []
for lo, hi in BANDS:
    a_med, a_n = median_in_band([(c, ips, it) for c, ips, it in adapt_ips], lo, hi)
    v_med, v_n = median_in_band([(c, ips, it) for c, ips, it in v1_ips], lo, hi)
    ratio = (a_med / v_med) if (a_med and v_med) else None
    rows.append([band_label(lo, hi),
                 round(a_med, 6) if a_med else "",
                 a_n,
                 round(v_med, 6) if v_med else "",
                 v_n,
                 round(ratio, 4) if ratio else ""])
    band_summary.append({"band": band_label(lo, hi),
                         "adaptive_median_ips": round(a_med, 6) if a_med else None,
                         "n_adaptive": a_n,
                         "v1_median_ips": round(v_med, 6) if v_med else None,
                         "n_v1": v_n,
                         "speedup": round(ratio, 4) if ratio else None})
write_csv(os.path.join(DP, "fig6_speedup_bands.csv"),
          ["cell_band", "adaptive_median_ips", "n_adaptive",
           "v1_median_ips", "n_v1", "speedup"], rows)

valid_ratios = [b for b in band_summary if b["speedup"] is not None]
# dip = local minimum among matched bands
dips = []
for i, b in enumerate(valid_ratios):
    prev = valid_ratios[i - 1]["speedup"] if i > 0 else None
    nxt = valid_ratios[i + 1]["speedup"] if i + 1 < len(valid_ratios) else None
    if prev and nxt and b["speedup"] < prev and b["speedup"] < nxt:
        dips.append({"band": b["band"], "speedup": b["speedup"]})
summary["fig6_speedup_bands"] = {
    "bands": band_summary,
    "matched_bands": [b["band"] for b in valid_ratios],
    "speedup_min": round(min(b["speedup"] for b in valid_ratios), 4),
    "speedup_max": round(max(b["speedup"] for b in valid_ratios), 4),
    "dips": dips,
}

# ============================================================================
# FIGURE 7 - adaptive phase-time shares (cells > 100); v1 not available
# ============================================================================
act = [d for d in adiag if d["cells"] > 100]
mean_mesh = st.mean(d["mesh"] for d in act)
mean_contact = st.mean(d["contact"] for d in act)
mean_polar = st.mean(d["polar"] for d in act)
mean_timeint = st.mean(d["timeint"] for d in act)
mean_total = st.mean(d["total"] for d in act)
sum4 = mean_mesh + mean_contact + mean_polar + mean_timeint
# renormalize the 4 named phases to 100 (matches caption methodology)
shares_renorm = {
    "mesh_refinement": round(100 * mean_mesh / sum4, 2),
    "contact_detection": round(100 * mean_contact / sum4, 2),
    "polarization_internal_forces": round(100 * mean_polar / sum4, 2),
    "time_integration": round(100 * mean_timeint / sum4, 2),
}
shares_vs_total = {
    "mesh_refinement": round(100 * mean_mesh / mean_total, 2),
    "contact_detection": round(100 * mean_contact / mean_total, 2),
    "polarization_internal_forces": round(100 * mean_polar / mean_total, 2),
    "time_integration": round(100 * mean_timeint / mean_total, 2),
}
# how much of total is untracked overhead (here ~0: total == sum4)
max_overhead = max((d["total"] - (d["mesh"] + d["contact"] + d["polar"] + d["timeint"]))
                   for d in act)

rows = [
    ["mesh_refinement", round(mean_mesh, 3), shares_renorm["mesh_refinement"], shares_vs_total["mesh_refinement"]],
    ["contact_detection", round(mean_contact, 3), shares_renorm["contact_detection"], shares_vs_total["contact_detection"]],
    ["polarization_internal_forces", round(mean_polar, 3), shares_renorm["polarization_internal_forces"], shares_vs_total["polarization_internal_forces"]],
    ["time_integration", round(mean_timeint, 3), shares_renorm["time_integration"], shares_vs_total["time_integration"]],
]
write_csv(os.path.join(DP, "fig7_phase_shares.csv"),
          ["phase", "mean_ms", "share_pct_renorm", "share_pct_vs_total"], rows)

summary["fig7_phase_shares"] = {
    "v1_available": False,
    "v1_note": "genuine v1.0 emits no per-phase *_ms timing in the new run",
    "n_rows_cells_gt_100": len(act),
    "adaptive_shares_renorm_pct": shares_renorm,
    "adaptive_shares_vs_total_pct": shares_vs_total,
    "max_untracked_overhead_ms": round(max_overhead, 3),
}

# ============================================================================
# PHYSICS CHECK - adaptive vs v1 mean_pressure at matched iterations
# ============================================================================
def load_bio(path):
    d = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            d[int(row["iteration"])] = {
                "pressure": float(row["mean_pressure"]),
                "cells": float(row["cell_count"]),
            }
    return d

ad_bio = load_bio(ADBIO)
v1_bio = load_bio(V1BIO)
common = sorted(set(ad_bio) & set(v1_bio))
rows = []
rel_devs = []
for it in common:
    pa = ad_bio[it]["pressure"]
    pv = v1_bio[it]["pressure"]
    denom = abs(pv) if abs(pv) > 1e-12 else float("nan")
    rel = abs(pa - pv) / denom if denom == denom else float("nan")
    rows.append([it, ad_bio[it]["cells"], v1_bio[it]["cells"],
                 round(pa, 6), round(pv, 6),
                 round(rel * 100, 4) if rel == rel else ""])
    if rel == rel:
        rel_devs.append(rel * 100)
write_csv(os.path.join(DP, "physics_check.csv"),
          ["iteration", "adaptive_cells", "v1_cells",
           "adaptive_mean_pressure", "v1_mean_pressure", "rel_dev_pct"], rows)

summary["physics_check"] = {
    "metric": "mean_pressure (adaptive vs v1) at matched iterations",
    "source": "plots/updated/cleaned_data/{adaptive,v1}_biological_cleaned.csv",
    "n_matched_iterations": len(rel_devs),
    "median_rel_dev_pct": round(st.median(rel_devs), 4),
    "mean_rel_dev_pct": round(st.mean(rel_devs), 4),
    "p90_rel_dev_pct": round(sorted(rel_devs)[int(0.9 * len(rel_devs))], 4),
    "max_rel_dev_pct": round(max(rel_devs), 4),
}

# ============================================================================
# HEADLINE
# ============================================================================
summary["headline"] = {
    "adaptive_final_cells": ADAPTIVE_FINAL_CELLS,
    "v1_concurrent_final_cells": V1_CONCURRENT_CELLS,
    "ratio_matched_wallclock": round(ADAPTIVE_FINAL_CELLS / V1_CONCURRENT_CELLS, 4),
    "adaptive_cov_max": round(max(cov_all), 4),
    "adaptive_imbalance_max": round(max(imb_all), 4),
    "phase_transition_iter": trans["iteration"] if trans else None,
    "phase_transition_cells": trans["cells"] if trans else None,
    "pressure_median_dev_pct": summary["physics_check"]["median_rel_dev_pct"],
    "note_v1_solo_tail_excluded": "sim_v1/_solo_tail (~15654 cells) excluded; v1 endpoint is concurrent iter 20100 = 12851 cells",
    "note_adaptive_oom": "adaptive OOM-killed at ~98 GB after iter 21900 / 26534 cells; not a clean shared stopping point",
}

with open(os.path.join(OUT, "03_new_run_recreated_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("DONE. Wrote datapoints + summary.")
print(json.dumps(summary["headline"], indent=2))
