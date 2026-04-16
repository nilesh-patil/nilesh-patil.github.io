#!/usr/bin/env python3
"""
Rosetta-Stone extraction of the published Static scatter cloud from figure-1.svg
(thread_imbalance_pct) and figure-2.svg (cov).

PathCollection_2 = 208 pts = ADAPTIVE (we have its data from the 20260124 CSV).
PathCollection_1 = 137 pts = STATIC (data lost; recover by inverting the fitted
device<->data transform calibrated on the adaptive cloud).

x-axis is log10(cells); y-axis is linear (value). Verify recovered Static stats
against the committed caption numbers, then write the recovered points to CSV.
"""
import sys, re, math, csv, os
import statistics as st
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get(
    "SIMUCELL3D_RESULTS",
    "/Users/nilesh-patil/versioned-projects/Github-personal.tmp/simucell3d/docs/simulation_results",
)
ORIG = os.path.join(RESULTS, "parallel_benchmark_20260124_091702")
# read the PUBLISHED figure SVGs; write recovered Static clouds next to this script
IMG = os.environ.get(
    "SIMUCELL3D_IMG_OUT",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "images", "blog", "simucell3d")),
)
OUT = os.path.join(HERE, "datapoints")
XLINK = "{http://www.w3.org/1999/xlink}href"


def lt(t):
    return t.split('}')[-1]


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float('nan')
    return b, a, r2


def collect_uses(group):
    """Return list of (x, y) device coords for <use> markers under a group,
    accumulating any translate() transforms on the use or ancestors."""
    pts = []
    for u in group.iter():
        if lt(u.tag) != 'use':
            continue
        x = float(u.get('x', '0'))
        y = float(u.get('y', '0'))
        tr = u.get('transform', '')
        m = re.search(r'translate\(\s*([-\d.]+)[ ,]+([-\d.]+)', tr)
        if m:
            x += float(m.group(1)); y += float(m.group(2))
        pts.append((x, y))
    return pts


def get_collection(root, gid):
    for g in root.iter():
        if lt(g.tag) == 'g' and g.get('id', '') == gid:
            return collect_uses(g)
    return []


def load_adaptive(col):
    """(cells, value) for the 20260124 adaptive perf_diag, sorted by cells."""
    rows = []
    with open(os.path.join(ORIG, "sim_adaptive", "performance_diagnostics.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["cells"]), float(r[col])))
    return rows


def extract(svgname, col, committed):
    root = ET.parse(os.path.join(IMG, svgname)).getroot()
    adapt_dev = get_collection(root, "PathCollection_2")   # 208 adaptive
    static_dev = get_collection(root, "PathCollection_1")  # 137 static
    data = load_adaptive(col)
    print(f"\n===== {svgname} ({col}) =====")
    print(f"  adapt markers={len(adapt_dev)}  static markers={len(static_dev)}  adapt CSV rows={len(data)}")

    # ---- X transform: log10(cells) -> x_dev, via sorted pairing (x monotone) ----
    adapt_sorted = sorted(adapt_dev, key=lambda p: p[0])      # by device x
    data_sorted = sorted(data, key=lambda d: d[0])            # by cells
    n = min(len(adapt_sorted), len(data_sorted))
    lx = [math.log10(c) for c, v in data_sorted[:n] if c > 0]
    xd = [p[0] for p in adapt_sorted[:n]][-len(lx):]
    bx, ax, r2x = linfit(lx, xd)
    print(f"  X-fit (log10 cells -> x_dev): b={bx:.4f} a={ax:.4f} R2={r2x:.6f}")

    def dev_to_cells(xdev):
        return 10 ** ((xdev - ax) / bx)

    # ---- Y transform: recover each adapt point's cells from x_dev, lookup value ----
    by_cells = {}
    for c, v in data:
        by_cells.setdefault(c, []).append(v)
    pairs = []
    for (xd_, yd_) in adapt_dev:
        c_est = dev_to_cells(xd_)
        # nearest known cell
        ck = min(by_cells, key=lambda c: abs(c - c_est))
        v = st.mean(by_cells[ck])
        pairs.append((v, yd_))
    vy = [p[0] for p in pairs]
    yd = [p[1] for p in pairs]
    by_, ay_, r2y = linfit(vy, yd)
    print(f"  Y-fit (value -> y_dev): b={by_:.4f} a={ay_:.4f} R2={r2y:.6f}")

    def dev_to_val(ydev):
        return (ydev - ay_) / by_

    # ---- invert static cloud ----
    static = []
    for (xd_, yd_) in static_dev:
        c = dev_to_cells(xd_)
        v = dev_to_val(yd_)
        static.append((c, v))
    static.sort()
    vals = [v for c, v in static]
    print(f"  RECOVERED static: n={len(static)} cells[{static[0][0]:.0f}..{static[-1][0]:.0f}] "
          f"value mean/median/max = {st.mean(vals):.4f}/{st.median(vals):.4f}/{max(vals):.4f}")
    print(f"  COMMITTED static            mean/median/max = {committed}")

    # also recovered adaptive (sanity: should match CSV stats)
    return static, (bx, ax, by_, ay_, r2x, r2y)


def main():
    s1, _ = extract("figure-1.svg", "thread_imbalance_pct",
                    "16.40/16.10/21.20")
    s2, _ = extract("figure-2.svg", "cov",
                    "0.1640/0.1610/0.2120")
    # write recovered static clouds
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fig1_static_recovered.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cells", "thread_imbalance_pct"])
        w.writerows([[round(c, 1), round(v, 4)] for c, v in s1])
    with open(os.path.join(OUT, "fig2_static_recovered.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cells", "cov"])
        w.writerows([[round(c, 1), round(v, 5)] for c, v in s2])
    print("\nWrote fig1_static_recovered.csv + fig2_static_recovered.csv")


if __name__ == "__main__":
    main()
