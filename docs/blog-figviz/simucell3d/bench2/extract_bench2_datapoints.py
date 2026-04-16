#!/usr/bin/env python3
"""
Extract the bench2 datapoints behind the bench2-*.svg figures from a saved copy
of the claude.ai artifact "SimuCell3D Growth Reproduction" (692AJVP7XwgEL81LKGMMhi,
version 1795354920-9a8c, read 2026-11-22). Writes the CSVs in ./datapoints/.

Two kinds of source:
  * the embedded JSON blob <script id="b2data"> (growth, spi, mem): copied
    point for point;
  * hard-coded JS tables (M milestones, ROWS outcome matrix, EX exponents,
    panel text): transcribed below, and each transcription is checked against
    the HTML by exact substring match, so a typo fails the run.
The six matched-case timings come from the companion artifact "SimuCell3D v1.0
vs 2.x" (VqoFDcYbtPRasFDaAxq1oC, version 1795354921-eba3, table "Speed on
matched cases"); four of the six also appear in A1's ROWS and are checked there.

Usage (stdlib only; the HTML is untrusted data, so run isolated):
  python3 -I extract_bench2_datapoints.py <saved-artifact.html>
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.join(HERE, "datapoints")
A1 = "A1 SimuCell3D Growth Reproduction"
A2 = "A2 SimuCell3D v1.0 vs 2.x"

# run id -> (series, build code, run number) ; series names are the figure labels
RUNS = {
    "at1-dt25-b1-full": ("v1.0 + crash fixes", "B1", 1),
    "at1-dt25-b1-full-r2": ("v1.0 + crash fixes", "B1", 2),
    "at1-dt25-b1-full-r3": ("v1.0 + crash fixes", "B1", 3),
    "at1-dt25-b3-full": ("2.x dev builds", "B3", 1),
    "at1-dt25-b4-full": ("2.x dev builds", "B4", 1),
    "at1-dt25-b5-full": ("2.x dev builds", "B5", 1),
    "at1-dt25-b6-full": ("2.x dev builds", "B6", 1),
    "at1-dt25-b7-full": ("v2.1.0", "B7", 1),
    "at1-dt25-b7-full-r2": ("v2.1.0", "B7", 2),
    "at1-dt25-b7-full-r3": ("v2.1.0", "B7", 3),
}


def need(html, s):
    if s not in html:
        sys.exit(f"transcription check failed, not found in artifact: {s!r}")


def write(name, header, rows):
    os.makedirs(DP, exist_ok=True)
    with open(os.path.join(DP, name), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote datapoints/{name} ({len(rows)} rows)")


def main(path):
    html = open(path, encoding="utf8").read()
    m = re.search(r'<script id="b2data" type="application/json">(.*?)</script>', html, re.S)
    b2 = json.loads(m.group(1))

    # ---- setup facts used in subtitles ---------------------------------------
    q_run = "Every long T1 run, 8 threads on c2-standard-16. Each run stops at 150k cells, the 48 h cap, or its own failure."
    q_cfg = "bench2’s growing aggregate (T1) is the lab’s benchmark file: contact model 1"
    q_dt = "so bench2 compares builds at dt 2.5e-8"
    for s in (q_run, q_cfg, q_dt):
        need(html, s)
    p = f"{A1}: bench2 scaling panel text"
    write("bench2_setup.csv", ["key", "value", "source"], [
        ("machine", "c2-standard-16", f"{p} '{q_run}'"),
        ("threads", 8, f"{p} '{q_run}'"),
        ("case", "T1 growing aggregate", f"{p} '{q_cfg}'"),
        ("contact_model", 1, f"{p} '{q_cfg}'"),
        ("dt", "2.5e-8", f"{p} '{q_dt}'"),
        ("stop_cells", "150k", f"{p} '{q_run}'"),
        ("time_cap_hours", 48, f"{p} '{q_run}'"),
    ])

    # ---- reach: final T1 cells per build at dt 2.5e-8 -----------------------
    for s in ["['dt 2.5e-8 (48 h)', ab('127 cells'), ab('22.7k–42.3k cells', 'abort 3/3'), cr('63 cells')",
              "['B1 r1 / r2 / r3', 'B1', ['2.50 / 2.60 / 2.49', '— / — / 10.81', null, null, null], 'abort at 25.1k / 22.7k / 42.3k']",
              "'135,773 (cap)'", "'138,135 (cap)'", "'136,215 (cap)'", "'143,394 (cap)'",
              "'144,919 (cap)'", "'148,914 (cap)'", "'142,314 (cap)'"]:
        need(html, s)
    src_rows, src_m = f"{A1}: JS ROWS, row 'dt 2.5e-8 (48 h)'", f"{A1}: JS M, column 'Final cells'"
    reach = [
        (1, "v1.0", "B0", 1, 127, "127", "abort", src_rows),
        (2, "v1.0 + crash fixes", "B1", 1, 25100, "25.1k", "abort", src_m),
        (3, "v1.0 + crash fixes", "B1", 2, 22700, "22.7k", "abort", src_m),
        (4, "v1.0 + crash fixes", "B1", 3, 42300, "42.3k", "abort", src_m),
        (5, "v2.0", "B2", 1, 63, "63", "crash", src_rows),
        (6, "2.x dev builds", "B3", 1, 135773, "135,773", "48 h cap", src_m),
        (7, "2.x dev builds", "B4", 1, 138135, "138,135", "48 h cap", src_m),
        (8, "2.x dev builds", "B5", 1, 136215, "136,215", "48 h cap", src_m),
        (9, "2.x dev builds", "B6", 1, 143394, "143,394", "48 h cap", src_m),
        (10, "v2.1.0", "B7", 1, 144919, "144,919", "48 h cap", src_m),
        (11, "v2.1.0", "B7", 2, 148914, "148,914", "48 h cap", src_m),
        (12, "v2.1.0", "B7", 3, 142314, "142,314", "48 h cap", src_m),
    ]
    write("bench2_reach.csv",
          ["row", "series", "build", "run", "final_cells", "final_cells_verbatim", "outcome", "source"],
          reach)

    # ---- growth: cells vs wall-clock hours, dt 2.5e-8 -----------------------
    rows = []
    for rid, (series, b, r) in RUNS.items():
        g = b2["growth"][rid]
        for h, c, it in g["pts"]:
            rows.append(("curve", series, b, r, h, c, it, "", f"{A1}: b2data.growth['{rid}'].pts (src {g['src']})"))
    # the three v1.0 + crash fixes aborts: last curve point, labelled with the M-table final
    for rid, lab in [("at1-dt25-b1-full", "25.1k"), ("at1-dt25-b1-full-r2", "22.7k"),
                     ("at1-dt25-b1-full-r3", "42.3k")]:
        series, b, r = RUNS[rid]
        h, c, it = b2["growth"][rid]["pts"][-1]
        rows.append(("abort_marker", series, b, r, h, c, it, lab,
                     f"{A1}: last point of b2data.growth['{rid}'].pts; label from JS M 'abort at 25.1k / 22.7k / 42.3k'"))
    need(html, "Each run stops at 150k cells, the 48 h cap, or its own failure.")
    rows.append(("time_cap", "", "", "", 48, "", "", "48 h cap", f"{A1}: bench2 scaling panel text 'the 48 h cap'"))
    write("bench2_growth.csv",
          ["kind", "series", "build", "run", "hours", "cells", "iteration", "label", "source"], rows)

    # ---- seconds per iteration vs cells -------------------------------------
    rows = []
    for rid, (series, b, r) in RUNS.items():
        for c, s in b2["spi"][rid]:
            rows.append(("curve", series, b, r, c, s, "", f"{A1}: b2data.spi['{rid}']"))
    need(html, "Per-iteration cost grows as N<sup>1.08–1.16</sup>.")
    need(html, "['B7 r1 / r2 / r3', 'B7', '1.10 / 1.09 / 1.08', '1.33 / 1.32 / 1.32']")
    for k in (1.08, 1.16):
        rows.append(("reference_exponent", "", "", "", "", "", k,
                     f"{A1}: Scaling exponents panel text 'Per-iteration cost grows as N^1.08-1.16'"))
    write("bench2_sec_per_iter.csv",
          ["kind", "series", "build", "run", "cells", "sec_per_iter", "exponent", "source"], rows)

    # ---- matched cases, hours, v1.0 + crash fixes (B1) vs v2.1.0 (B7) -------
    for s in ["T1 to 10k: <b>1.28 h vs 2.49 h</b>", "ok('6.28 h')", "ok('3.44 h; same geometry in all builds')",
              "ok('22.1 / 20.6 / 14.6 h')", "ok('8.6 / 8.6 / 8.1 h')"]:
        need(html, s)
    src2 = f"{A2}: table 'Speed on matched cases'"
    rows = [
        (1, "Growing aggregate to 10k cells", "A", 2.49, 2.49, 1.28, 1.28, "1.9", src2 + f"; also {A1} verdict tile"),
        (2, "Vesicle, eps 2.0", "A", 16.0, 16.8, 6.2, 7.0, "2.5", src2 + " (A1 has no copy of this row)"),
        (3, "Tube", "A", 6.28, 6.28, 3.44, 3.44, "1.8", src2 + f"; also {A1} JS ROWS 'T4 tube Arm A'"),
        (4, "Vesicle, eps 2.0", "B", 22.1, 22.1, 8.6, 8.6, "2.6", src2 + f"; also {A1} JS ROWS 'T2, T3 multi, T4'"),
        (5, "Vesicle multilayer", "B", 20.6, 20.6, 8.6, 8.6, "2.4", src2 + f"; also {A1} JS ROWS 'T2, T3 multi, T4'"),
        (6, "Tube", "B", 14.6, 14.6, 8.1, 8.1, "1.8", src2 + f"; also {A1} JS ROWS 'T2, T3 multi, T4'"),
    ]
    write("bench2_matched_cases.csv",
          ["row", "case", "arm", "v1_fix_hours_lo", "v1_fix_hours_hi", "v210_hours_lo", "v210_hours_hi",
           "ratio_verbatim", "source"], rows)

    # ---- memory: resident GiB vs cells, dt 2.5e-8 runs -----------------------
    rows = []
    for run, b, cells, col4, gib in b2["mem"]:
        if "dt25" not in run:
            continue
        rows.append(("sample", run, b, cells, col4, gib, f"{A1}: b2data.mem"))
    need(html, "hit the 61.6 GiB cap; B6/B7 stay at ~0.13 GB per 1k cells.")
    rows.append(("memory_cap", "", "", "", "", 61.6, f"{A1}: cB2Mem panel text and '_cap' line"))
    rows.append(("v210_slope_gib_per_1k_cells", "", "", "", "", 0.13, f"{A1}: cB2Mem panel text '~0.13 GB per 1k cells'"))
    write("bench2_memory.csv",
          ["kind", "run", "build", "cells", "col4_unlabeled", "rss_gib", "source"], rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
