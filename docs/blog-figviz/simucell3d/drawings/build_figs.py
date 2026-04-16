#!/usr/bin/env python3
"""Build the seven inline SVG line drawings for the SimuCell3D post (laptop-first, 720 wide).
Run: python3 build_figs.py           (writes out/<id>.svg; needs numpy)"""
import heapq
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kit import (Svg, MONO, f, line, path, poly_d, rect, circle, text, arrow, hatch_d, blob,
                 out_normal, scale_pts, dots, clip_convex)
from mesh3d import Cell

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
W = 720
MAIN, SEC = 16, 14          # label sizes; nothing below 13
DIM = 0.78                  # opacity of secondary text


def sub(base, s, size=13):
    """base with a subscript, as tspans (baseline-shift is not portable)."""
    return (f'{base}<tspan dy="4" font-size="{size}" font-style="normal">{s}</tspan>'
            f'<tspan dy="-4" font-size="{size}"> </tspan>')


def it(s):
    return f'<tspan font-style="italic">{s}</tspan>'


def leader(x1, y1, x2, y2, dot=True):
    out = [line(x1, y1, x2, y2, stroke="currentColor", stroke_width=1.0, fill="none", opacity=0.6)]
    if dot:
        out.append(circle(x2, y2, 2.2, fill="currentColor", stroke="none", opacity=0.75))
    return out


# ---------------------------------------------------------------------------
# fig-mesh
# ---------------------------------------------------------------------------
def fig_mesh():
    H = 380
    s = Svg("fig-mesh", W, H, "One cell as a closed surface of triangles",
            "A rounded cell drawn as a closed mesh of triangles, with one node, one edge and one "
            "face labelled. A second, fainter cell is pressed flat against its right side.")
    A = Cell(300, 166, (142, 132, 136), seed=7, clamp=[(1, 112)])
    B = Cell(506, 172, (114, 117, 114), seed=12, clamp=[(-1, -86)])
    s.open(opacity=0.5)
    s.add(B.draw(node_r=2.8, back=False, sw_in=1.2, sw_rim=1.8))
    s.close()
    ni = A.nearest_node(224, 70)
    ei = A.nearest_inner_edge(207, 177)
    fi = A.nearest_face(232, 262)
    s.add(A.draw(node_r=3.4, sw_in=1.2, sw_rim=1.9))
    fx, fy = A.P[list(A.F[fi])].mean(axis=0)
    s.add(path(poly_d([tuple(A.P[i]) for i in A.F[fi]]), fill="currentColor", fill_opacity=0.24,
               stroke="currentColor", stroke_width=1.5))
    (ax, ay), (bx, by) = A.P[ei[0]], A.P[ei[1]]
    s.add(line(ax, ay, bx, by, stroke="currentColor", stroke_width=3.4, fill="none",
               stroke_linecap="round"))
    nx, ny = A.P[ni]
    s.add(circle(nx, ny, 5.8, fill="currentColor", stroke="none"))
    lx = 82
    s.add(leader(lx, 54, nx - 8, ny - 1.5, dot=False))
    s.add(leader(lx, 174, (ax + bx) / 2 - 6, (ay + by) / 2, dot=False))
    s.add(leader(lx, 294, fx, fy))
    # the contact leader ends on the flat interface between the two cells
    flat_a = max(p[0] for p in A.P)
    flat_b = min(p[0] for p in B.P)
    xi = (flat_a + flat_b) / 2
    top_a = min(p[1] for p in A.P if p[0] > flat_a - 0.5)
    top_b = min(p[1] for p in B.P if p[0] < flat_b + 0.5)
    yi = max(top_a, top_b) + 26
    s.add(leader(xi, 34, xi, yi - 3, dot=False))
    s.add(circle(xi, yi, 3, fill="currentColor", stroke="none"))
    s.text_group()
    s.add(text(28, 60, "node", 17, weight=600))
    s.add(text(28, 180, "edge", 17, weight=600))
    s.add(text(28, 300, "face", 17, weight=600))
    s.add(text(xi, 26, "contact", MAIN, anchor="middle"))
    s.add(text(590, 40, "a second cell", SEC, anchor="middle", opacity=DIM))
    s.add(text(300, 342, "one cell: a closed surface of triangles", MAIN, anchor="middle", weight=600))
    s.add(text(300, 364, "in the benchmark: about 120 nodes and 240 triangles", SEC, anchor="middle",
               opacity=DIM))
    s.close()
    s.write(os.path.join(OUT, "fig-mesh.svg"))


# ---------------------------------------------------------------------------
# fig-forces: five panels in one row
# ---------------------------------------------------------------------------
def spring_d(p, q, zigs=3, amp=3.6, lead=0.2):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    pts = [p, (p[0] + ux * L * lead, p[1] + uy * L * lead)]
    n = zigs * 2
    for k in range(n):
        t = lead + (1 - 2 * lead) * (k + 0.5) / n
        sgn = 1 if k % 2 == 0 else -1
        pts.append((p[0] + ux * L * t + px * amp * sgn, p[1] + uy * L * t + py * amp * sgn))
    pts += [(q[0] - ux * L * lead, q[1] - uy * L * lead), q]
    return poly_d(pts, close=False)


def superblob(cx, cy, rx, ry, n, power=0.55, phase=0.0):
    pts = []
    for i in range(n):
        t = 2 * math.pi * (i + phase) / n
        c, s_ = math.cos(t), math.sin(t)
        pts.append((cx + rx * math.copysign(abs(c) ** power, c), cy + ry * math.copysign(abs(s_) ** power, s_)))
    return pts


def fig_forces():
    H = 236
    s = Svg("fig-forces", W, H, "Five forces on the nodes of a cell",
            "Five numbered panels in a row, each a cut through a cell outline with arrows: "
            "1 pressure pushing every face outward, 2 area elasticity as springs in the surface, "
            "3 surface tension pulling along the surface, 4 bending flattening a sharp fold, and "
            "5 two cells that attract when close and repel when they overlap.")
    c1, c2, c3, c4 = 66, 182, 298, 414
    ca, cb = 534, 656
    y1 = 78
    body = dict(stroke="currentColor", stroke_width=1.5, fill="currentColor", fill_opacity=0.06,
                stroke_linejoin="round")

    # 1 pressure
    c = (c1, y1)
    P = blob(c[0], c[1], 27, 9, seed=4, jit=0.06, rot=0.3)
    s.add(path(poly_d(scale_pts(P, c, 1.7)), stroke="currentColor", stroke_width=1.1, fill="none",
               stroke_dasharray="4 4", opacity=0.55, stroke_linejoin="round"))
    s.add(path(poly_d(P), **body))
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        nx, ny = out_normal(p, q, c)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        s.add(arrow(mx + nx * 3.5, my + ny * 3.5, mx + nx * 14.5, my + ny * 14.5, head=6, sw=1.3))
    s.add(dots(P, 2.4, fill="currentColor", stroke="none"))

    # 2 area elasticity
    c = (c2, y1)
    P = blob(c[0], c[1], 42, 8, seed=9, jit=0.05, rot=0.2)
    s.add(path(poly_d(P), fill="currentColor", fill_opacity=0.06, stroke="none"))
    s.add(path(poly_d(scale_pts(P, c, 0.5)), stroke="currentColor", stroke_width=1.1, fill="none",
               stroke_dasharray="4 4", opacity=0.55, stroke_linejoin="round"))
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        s.add(path(spring_d(p, q), stroke="currentColor", stroke_width=1.3, fill="none",
                   stroke_linejoin="round"))
        if i % 2 == 0:
            dx, dy = c[0] - p[0], c[1] - p[1]
            L = math.hypot(dx, dy)
            s.add(arrow(p[0] + dx / L * 5, p[1] + dy / L * 5, p[0] + dx / L * 14.5, p[1] + dy / L * 14.5,
                        head=5.5, sw=1.3))
    s.add(dots(P, 2.6, fill="currentColor", stroke="none"))

    # 3 surface tension
    c = (c3, y1)
    P = blob(c[0], c[1], 38, 8, seed=21, jit=0.05, rot=0.45)
    s.add(path(poly_d(P), **body))
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        nx, ny = out_normal(p, q, c)
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        mx, my = (p[0] + q[0]) / 2 + nx * 7.5, (p[1] + q[1]) / 2 + ny * 7.5
        a = min(L / 2 - 1.5, 13.5)
        s.add(arrow(mx - ux * a, my - uy * a, mx - ux * 2.2, my - uy * 2.2, head=5.5, sw=1.2))
        s.add(arrow(mx + ux * a, my + uy * a, mx + ux * 2.2, my + uy * 2.2, head=5.5, sw=1.2))
    s.add(dots(P, 2.4, fill="currentColor", stroke="none"))

    # 4 bending
    c = (c4, y1 + 10)
    P = blob(c[0], c[1], 34, 9, seed=15, jit=0.04, rot=-math.pi / 2, ajit=0.04)
    k = 0
    P[k] = (c[0] + (P[k][0] - c[0]) * 1.75, c[1] + (P[k][1] - c[1]) * 1.75)
    pa, pb = P[k - 1], P[(k + 1) % len(P)]
    s.add(path(poly_d(P), **body))
    s.add(line(pa[0], pa[1], pb[0], pb[1], stroke="currentColor", stroke_width=1.1, fill="none",
               stroke_dasharray="4 4", opacity=0.55))
    s.add(path(poly_d([pa, P[k], pb], close=False), stroke="currentColor", stroke_width=2.4,
               fill="none", stroke_linejoin="round", stroke_linecap="round"))
    s.add(dots(P, 2.4, fill="currentColor", stroke="none"))
    s.add(arrow(P[k][0] + 12, P[k][1] - 4, P[k][0] + 12, P[k][1] + 13, head=6, sw=1.4))
    s.add(arrow(P[k][0] - 12, P[k][1] - 4, P[k][0] - 12, P[k][1] + 13, head=6, sw=1.4))
    for p, sg in ((pa, -1), (pb, 1)):
        s.add(arrow(p[0] + sg * 5, p[1] - 3, p[0] + sg * 5, p[1] - 16, head=5.5, sw=1.3))

    # 5 contact: attract (left pair), repel (right pair)
    cy = y1 + 2

    def pair(cx, half, rx=23, ry=30):
        return (superblob(cx - half, cy, rx, ry, 14, phase=0.0),
                superblob(cx + half, cy, rx, ry, 14, phase=0.5))

    La, Ra = pair(ca, 32.5)
    s.add(path(poly_d(La), **body))
    s.add(path(poly_d(Ra), **body))
    s.add(dots(La + Ra, 2.2, fill="currentColor", stroke="none"))
    for p in sorted(La, key=lambda p: -p[0])[:3]:
        s.add(arrow(p[0] + 3, p[1], p[0] + 11, p[1], head=5, sw=1.2))
    for p in sorted(Ra, key=lambda p: p[0])[:4]:
        if abs(p[1] - cy) < 25:
            s.add(arrow(p[0] - 3, p[1], p[0] - 11, p[1], head=5, sw=1.2))
    Lb, Rb = pair(cb, 19)
    ov = clip_convex(Lb, Rb)
    s.add(path(poly_d(Lb), **body))
    s.add(path(poly_d(Rb), **body))
    if ov:
        s.add(path(poly_d(ov), fill="currentColor", fill_opacity=0.3, stroke="none"))
    s.add(dots(Lb + Rb, 2.2, fill="currentColor", stroke="none"))
    for yy in (cy - 42, cy + 42):
        s.add(arrow(cb - 5, yy, cb - 22, yy, head=6, sw=1.4))
        s.add(arrow(cb + 5, yy, cb + 22, yy, head=6, sw=1.4))

    # labels on a shared four-row grid
    r0, r1, r2, r3 = 156, 177, 197, 216
    s.text_group()
    s.add(text(c1, y1 + 5, "V", 15, anchor="middle", italic=True))
    s.add(text(c1 + 45, y1 - 46, sub("V", "0"), 15, anchor="middle", italic=True, opacity=0.85))
    s.add(text(c2, y1 + 5, sub("A", "0"), 15, anchor="middle", italic=True, opacity=0.85))
    s.add(text(c1, r0, "1 pressure, " + it("K"), MAIN, anchor="middle", weight=600))
    for y, t in ((r1, "pushes every"), (r2, "face out toward"), (r3, "a target volume")):
        s.add(text(c1, y, t, SEC, anchor="middle", opacity=DIM))
    s.add(text(c2, r0, "2 area", MAIN, anchor="middle", weight=600))
    s.add(text(c2, r1, "elasticity, " + it(sub("k", "a")), MAIN, anchor="middle", weight=600))
    for y, t in ((r2, "a spring around"), (r3, "a target area")):
        s.add(text(c2, y, t, SEC, anchor="middle", opacity=DIM))
    s.add(text(c3, r0, "3 surface", MAIN, anchor="middle", weight=600))
    s.add(text(c3, r1, "tension, " + it("γ"), MAIN, anchor="middle", weight=600))
    for y, t in ((r2, "always shrinks"), (r3, "the surface")):
        s.add(text(c3, y, t, SEC, anchor="middle", opacity=DIM))
    s.add(text(c4, r0, "4 bending, " + it(sub("k", "b")), MAIN, anchor="middle", weight=600))
    for y, t in ((r1, "resists a sharp"), (r2, "fold between"), (r3, "two faces")):
        s.add(text(c4, y, t, SEC, anchor="middle", opacity=DIM))
    cm = (ca + cb) / 2
    s.add(text(cm, r0, "5 contact", MAIN, anchor="middle", weight=600))
    s.add(text(cm, r1, "between two cells", MAIN, anchor="middle", weight=600))
    for y, t in ((r2, "inside the"), (r3, "cut-off: attract")):
        s.add(text(ca, y, t, SEC, anchor="middle", opacity=DIM))
    for y, t in ((r2, "on overlap:"), (r3, "repel")):
        s.add(text(cb, y, t, SEC, anchor="middle", opacity=DIM))
    s.close()
    s.write(os.path.join(OUT, "fig-forces.svg"))


# ---------------------------------------------------------------------------
# fig-divide
# ---------------------------------------------------------------------------
def fig_divide():
    H = 318
    s = Svg("fig-divide", W, H, "A cell grows, divides and is remeshed",
            "Three steps from left to right: a cell mesh swelling toward a larger dashed target "
            "outline, an elongated cell cut by a plane through its centre across its longest axis, "
            "and two daughter cells each with a fresh triangle mesh.")
    xs = (118, 360, 602)
    cy = 128
    th = -0.36
    ux, uy = math.cos(th), math.sin(th)
    px, py = -uy, ux
    kw = dict(node_r=2.7, back=False, sw_in=1.2, sw_rim=1.8)
    # 1 grow
    A = Cell(xs[0], cy, (52, 52, 52), seed=3)
    s.add(circle(xs[0], cy, 76, stroke="currentColor", stroke_width=1.1, fill="none",
                 stroke_dasharray="5 5", opacity=0.55))
    s.add(A.draw(**kw))
    for a in (40, 140, 220, 320):
        r = math.radians(a)
        s.add(arrow(xs[0] + 58 * math.cos(r), cy + 58 * math.sin(r), xs[0] + 72 * math.cos(r),
                    cy + 72 * math.sin(r), head=6.5, sw=1.4))
    # 2 divide
    M = Cell(xs[1], cy, (80, 53, 53), seed=5, theta=th)
    s.add(line(xs[1] - ux * 106, cy - uy * 106, xs[1] + ux * 106, cy + uy * 106, stroke="currentColor",
               stroke_width=1.1, fill="none", stroke_dasharray="9 3 1.5 3", opacity=0.6))
    s.add(M.draw(**kw))
    s.open_accent()
    s.add(line(xs[1] - px * 77, cy - py * 77, xs[1] + px * 77, cy + py * 77, stroke="currentColor",
               stroke_width=2.6, fill="none", stroke_linecap="round"))
    s.close()
    # 3 two daughters
    d = 36
    D1 = Cell(xs[2] - ux * d, cy - uy * d, (43, 53, 53), seed=21, clamp=[(1, 29)], theta=th)
    D2 = Cell(xs[2] + ux * d, cy + uy * d, (43, 53, 53), seed=34, clamp=[(-1, -29)], theta=th)
    s.add(D1.draw(**kw))
    s.add(D2.draw(**kw))
    s.add(arrow(208, cy, 242, cy, head=8, sw=1.6, opacity=0.7))
    s.add(arrow(478, cy, 512, cy, head=8, sw=1.6, opacity=0.7))
    s.text_group()
    s.add(text(xs[0], cy - 86, "target volume", SEC, anchor="middle", opacity=DIM))
    s.add(text(xs[1] + ux * 106 + 6, cy + uy * 106 - 9, "longest axis", SEC, anchor="middle", opacity=DIM))
    ty = 246
    for x, n, lab in zip(xs, "123", ("grow", "divide", "remesh")):
        s.add(text(x, ty, f"{n}  {lab}", 17, anchor="middle", weight=600))
    rows = (ty + 22, ty + 41, ty + 60)
    for x, lines_ in ((xs[0], ("the target volume rises", "and the cell follows")),
                      (xs[1], ("past the division volume, a plane", "cuts through the centre,", "across the longest axis"))):
        for y, t in zip(rows, lines_):
            s.add(text(x, y, t, SEC, anchor="middle", opacity=DIM))
    s.add(text(xs[2], rows[0], "two daughters, each with a new", SEC, anchor="middle", opacity=DIM))
    s.add(text(xs[2], rows[1], "mesh; every edge stays between", SEC, anchor="middle", opacity=DIM))
    s.add(text(xs[2], rows[2], it(sub("l", "min")) + "and 3 " + it(sub("l", "min")), SEC,
               anchor="middle", opacity=DIM))
    s.close()
    s.open_accent()
    s.text_group()
    s.add(text(xs[1] - px * 77 - 6, cy - py * 77 - 10, "division plane", SEC + 1, anchor="middle", weight=600))
    s.close()
    s.close()
    s.write(os.path.join(OUT, "fig-divide.svg"))


# ---------------------------------------------------------------------------
# fig-iteration: time left to right, 8 thread rows, phases as columns
# ---------------------------------------------------------------------------
def fig_iteration():
    H = 371
    s = Svg("fig-iteration", W, H, "One iteration, phase by phase",
            "A strip of eight thread rows with time running left to right, cut into seven phases "
            "in order. Divide, reset face types, refine meshes, contact and polarization plus "
            "internal forces fill all eight rows; time integration and remove cells fill one row. "
            "Contact takes three quarters of the width and starts with a short grid build on one row.")
    x0, x1 = 88, 706
    n, rh, rg = 8, 15, 3
    yT = 90
    yB = yT + n * rh + (n - 1) * rg
    g = 3
    ws = (x1 - x0 - 6 * g) / 24.0        # six short phases + contact (= 18 short widths)
    wc = 18 * ws
    names = ["divide", "reset face types", "refine meshes", "contact",
             "polarization + internal forces", "time integration", "remove cells"]
    par = [True, True, True, True, True, False, False]
    cols, x = [], x0
    for nm in names:
        w = wc if nm == "contact" else ws
        cols.append((x, w))
        x += w + g

    def col_rects(xa, w, busy_rows):
        busy, idle = [], []
        for k in range(n):
            r = rect(xa, yT + k * (rh + rg), w, rh, rx=2)
            (busy if k < busy_rows else idle).append(r)
        return busy, idle

    idle_all = []
    s.open(stroke="currentColor", stroke_width=1.1, fill="currentColor", fill_opacity=0.09)
    for i, (xa, w) in enumerate(cols):
        if names[i] == "contact":
            continue
        busy, idle = col_rects(xa, w, n if par[i] else 1)
        s.add(busy)
        idle_all += idle
    s.close()
    # contact: a short serial grid build, then all rows
    cx0, cw = cols[3]
    gw = ws + 1
    s.open_accent(stroke="currentColor", stroke_width=1.2, fill="currentColor", fill_opacity=0.2)
    busy, idle_c = col_rects(cx0, gw, 1)
    s.add(busy)
    busy, _ = col_rects(cx0 + gw + g, cw - gw - g, n)
    s.add(busy)
    s.close()
    s.open(stroke="currentColor", stroke_width=0.9, fill="none", opacity=0.34, stroke_dasharray="2.5 2.5")
    s.add(idle_all)
    s.close()
    s.open_accent(stroke="currentColor", stroke_width=0.9, fill="none", opacity=0.5,
                  stroke_dasharray="2.5 2.5")
    s.add(idle_c)
    s.close()
    # leaders: first three phases up, last three down
    up_rows = (28, 50, 72)
    dn_rows = (yB + 26, yB + 48, yB + 70)
    lead = []
    for i, yb in zip((0, 1, 2), up_rows):
        xm = cols[i][0] + cols[i][1] / 2
        lead.append(f"M{f(xm)},{f(yT - 3)} L{f(xm)},{f(yb - 5)} L{f(xm + 7)},{f(yb - 5)}")
    for i, yb in zip((4, 5, 6), dn_rows):
        xm = cols[i][0] + cols[i][1] / 2
        lead.append(f"M{f(xm)},{f(yB + 3)} L{f(xm)},{f(yb - 5)} L{f(xm - 7)},{f(yb - 5)}")
    s.add(path(" ".join(lead), stroke="currentColor", stroke_width=1.0, fill="none", opacity=0.6))
    # thread bracket, time axis
    s.add(path(f"M{f(x0 - 6)},{f(yT)} L{f(x0 - 10)},{f(yT)} L{f(x0 - 10)},{f(yB)} L{f(x0 - 6)},{f(yB)}",
               stroke="currentColor", stroke_width=1.0, fill="none", opacity=0.6))
    ya = yB + 90
    s.add(arrow(x0, ya, x1, ya, head=7, sw=1.1, opacity=0.6))

    def lab(nm, threaded):
        mark = "all threads" if threaded else "one thread"
        return (f'<tspan font-weight="600">{nm}</tspan>'
                f'<tspan dx="9" font-size="{SEC}" fill-opacity="{DIM}">{mark}</tspan>')

    s.text_group()
    s.add(text(8, (yT + yB) / 2 + 5, "8 threads", SEC, opacity=DIM))
    for i, yb in zip((0, 1, 2), up_rows):
        xm = cols[i][0] + cols[i][1] / 2
        if names[i] == "divide":
            s.add(text(xm + 11, yb, '<tspan font-weight="600">divide</tspan>'
                       f'<tspan dx="6" font-size="{SEC}">(every 5 iterations)</tspan>'
                       f'<tspan dx="9" font-size="{SEC}" fill-opacity="{DIM}">all threads</tspan>', MAIN))
            continue
        s.add(text(xm + 11, yb, lab(names[i], par[i]), MAIN))
    for i, yb in zip((4, 5, 6), dn_rows):
        xm = cols[i][0] + cols[i][1] / 2
        s.add(text(xm - 11, yb, lab(names[i], par[i]), MAIN, anchor="end"))
    s.add(text(x1, ya + 19, "time", SEC, anchor="end", opacity=DIM))
    s.add(text(x0, ya + 19, "short phases and the grid build are drawn equal, not to scale", SEC,
               opacity=DIM))
    s.add(text(x0, ya + 38, "not shown: a statistics pass every 50 iterations, on one thread", SEC,
               opacity=DIM))
    s.close()
    # contact labels in the accent
    xk = 430
    xg = cx0 + gw / 2
    s.open_accent()
    s.add(path(f"M{f(xk)},{f(yT - 3)} L{f(xk)},{f(up_rows[0] - 5)} L{f(xk + 7)},{f(up_rows[0] - 5)} "
               f"M{f(xg)},{f(yB + 3)} L{f(xg)},{f(dn_rows[1] - 5)} L{f(xg + 7)},{f(dn_rows[1] - 5)}",
               stroke="currentColor", stroke_width=1.1, fill="none"))
    s.text_group()
    s.add(text(xk + 11, up_rows[0], '<tspan font-weight="700">contact</tspan>'
               f'<tspan dx="9" font-size="{SEC}">all threads</tspan>', 17))
    s.add(text(xk + 11, up_rows[1], "about three quarters of an", SEC))
    s.add(text(xk + 11, up_rows[2], "iteration at 100,000 cells", SEC))
    s.add(text(xg + 11, dn_rows[1], "grid build: one thread", SEC, weight=600))
    s.close()
    s.close()
    s.write(os.path.join(OUT, "fig-iteration.svg"))


# ---------------------------------------------------------------------------
# fig-grid: a landscape crop of a tissue edge on a voxel grid
# ---------------------------------------------------------------------------
def clip_half(poly, px, py, nx, ny):
    """Keep the part of poly where (p - P).n <= 0."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da = (a[0] - px) * nx + (a[1] - py) * ny
        db = (b[0] - px) * nx + (b[1] - py) * ny
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def chaikin(poly, it_=2):
    for _ in range(it_):
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        poly = out
    return poly


def resample(poly, spacing, rng):
    n = len(poly)
    seg = [math.hypot(poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1]) for i in range(n)]
    total = sum(seg)
    m = max(8, round(total / spacing))
    stops = sorted(((k + rng.uniform(-0.22, 0.22)) / m) % 1.0 * total for k in range(m))
    out, acc, i = [], 0.0, 0
    for st in stops:
        while acc + seg[i] < st:
            acc += seg[i]
            i += 1
        t = (st - acc) / seg[i]
        a, b = poly[i], poly[(i + 1) % n]
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def clip_seg(a, b, xa, ya, xb, yb):
    """Liang-Barsky: the part of segment a-b inside the rectangle, or None."""
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - xa), (dx, xb - a[0]), (-dy, a[1] - ya), (dy, yb - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t1 - t0 < 1e-9:
        return None
    return ((a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy))


def fig_grid():
    vs, cols, nrows = 42, 11, 7
    gx0, gy0 = 12, 12
    gw, gh = cols * vs, nrows * vs
    gx1, gy1 = gx0 + gw, gy0 + gh
    H = gy1 + 70
    s = Svg("fig-grid", W, H, "Finding contacts with a grid of voxels",
            "A slice through the edge of a tissue laid over a square grid of voxels, with labels "
            "to the right. One highlighted voxel deep inside the tissue holds a node and many faces "
            "of other cells, drawn thick. One highlighted voxel on the free surface holds a node "
            "and no faces of other cells.")
    rng = random.Random(11)
    d = 98
    centres = []
    for j in range(-1, 6):
        yy = gy0 - 22 + j * d * 0.866
        for i in range(-1, 7):
            xx = gx0 - 44 + i * d + (d / 2 if j % 2 else 0)
            if xx < gx0 + 296 + 26 * math.sin((yy - gy0) / 62 + 0.4):
                centres.append((xx + rng.uniform(-5, 5), yy + rng.uniform(-5, 5)))
    radii = [57 + rng.uniform(-3, 3) for _ in centres]
    cells = []
    for i, (cx, cy) in enumerate(centres):
        poly = [(cx + radii[i] * math.cos(2 * math.pi * k / 60), cy + radii[i] * math.sin(2 * math.pi * k / 60))
                for k in range(60)]
        for j, (ox, oy) in enumerate(centres):
            if j == i:
                continue
            dist = math.hypot(ox - cx, oy - cy)
            if dist > radii[i] + radii[j]:
                continue
            nx, ny = (ox - cx) / dist, (oy - cy) / dist
            cut = dist * radii[i] / (radii[i] + radii[j]) - 2.4
            poly = clip_half(poly, cx + nx * cut, cy + ny * cut, nx, ny)
        poly = chaikin(poly, 2)
        cells.append(resample(poly, 14.5, rng))

    def inside(p, m=0.0):
        return gx0 + m <= p[0] <= gx1 - m and gy0 + m <= p[1] <= gy1 - m

    def vox(p):
        return (int((p[0] - gx0) // vs), int((p[1] - gy0) // vs))

    binned = {}
    for ci, nodes in enumerate(cells):
        m = len(nodes)
        for k in range(m):
            a, b = nodes[k], nodes[(k + 1) % m]
            if clip_seg(a, b, gx0, gy0, gx1, gy1) is None:
                continue
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            for v in {vox(a), vox(b), vox(mid)}:
                binned.setdefault(v, []).append((ci, k))

    def others(v, ci):
        return [(c, k) for c, k in binned.get(v, []) if c != ci]

    def frac(p, v):
        return (p[0] - gx0) / vs - v[0], (p[1] - gy0) / vs - v[1]

    # deep node: bottom row of the crop, far from the free surface, most faces of other cells
    best = None
    for ci, nodes in enumerate(cells):
        for p in nodes:
            if not inside(p, 2):
                continue
            v = vox(p)
            fx, fy = frac(p, v)
            if v[1] != nrows - 1 or not (1 <= v[0] <= 4) or not (0.25 < fx < 0.75 and 0.2 < fy < 0.75):
                continue
            sc = len(others(v, ci))
            if best is None or sc > best[0]:
                best = (sc, ci, p, v)
    _, dci, dnode, dv = best
    # surface node: nothing from other cells in its voxel, and only empty voxels to its right
    best = None
    for ci, nodes in enumerate(cells):
        for p in nodes:
            if not inside(p, 2):
                continue
            v = vox(p)
            fx, fy = frac(p, v)
            if not (2 <= v[1] <= 3) or others(v, ci) or not (0.25 < fx < 0.7 and 0.25 < fy < 0.75):
                continue
            if any((c, v[1]) in binned for c in range(v[0] + 1, cols)):
                continue
            if best is None or p[0] > best[0]:
                best = (p[0], ci, p, v)
    _, sci, snode, sv = best

    # grid
    g = []
    for i in range(cols + 1):
        g.append(f"M{f(gx0 + i * vs)},{f(gy0)} L{f(gx0 + i * vs)},{f(gy1)}")
    for j in range(nrows + 1):
        g.append(f"M{f(gx0)},{f(gy0 + j * vs)} L{f(gx1)},{f(gy0 + j * vs)}")
    s.add(path(" ".join(g), stroke="currentColor", stroke_width=0.8, fill="none", opacity=0.26))
    s.open_accent()
    for v in (dv, sv):
        s.add(rect(gx0 + v[0] * vs, gy0 + v[1] * vs, vs, vs, fill="currentColor", fill_opacity=0.14,
                   stroke="none"))
    s.close()
    # cells, cut at the edge of the crop
    fills, strokes, nodes_in = [], [], []
    for nodes in cells:
        poly = list(nodes)
        for (px_, py_, nx, ny) in ((gx0, 0, -1, 0), (gx1, 0, 1, 0), (0, gy0, 0, -1), (0, gy1, 0, 1)):
            poly = clip_half(poly, px_, py_, nx, ny) if poly else poly
        if len(poly) >= 3:
            fills.append(poly_d(poly))
        m = len(nodes)
        for k in range(m):
            c = clip_seg(nodes[k], nodes[(k + 1) % m], gx0, gy0, gx1, gy1)
            if c:
                strokes.append(f"M{f(c[0][0])},{f(c[0][1])} L{f(c[1][0])},{f(c[1][1])}")
        nodes_in += [p for p in nodes if inside(p, 1.5)]
    s.add(path(" ".join(fills), fill="currentColor", fill_opacity=0.055, stroke="none"))
    s.add(path(" ".join(strokes), stroke="currentColor", stroke_width=1.3, fill="none",
               stroke_linecap="round"))
    s.open(fill="currentColor", stroke="none", opacity=0.85)
    s.add(dots(nodes_in, 1.8))
    s.close()
    # accent: thick faces of other cells in the deep voxel, frames, nodes, leaders, labels
    lxr = gx1 + 16            # where the label column starts
    s.open_accent()
    segs = []
    for c, k in others(dv, dci):
        cl = clip_seg(cells[c][k], cells[c][(k + 1) % len(cells[c])], gx0, gy0, gx1, gy1)
        if cl:
            segs.append(f"M{f(cl[0][0])},{f(cl[0][1])} L{f(cl[1][0])},{f(cl[1][1])}")
    s.add(path(" ".join(segs), stroke="currentColor", stroke_width=3.4, fill="none", stroke_linecap="round"))
    for v in (dv, sv):
        s.add(rect(gx0 + v[0] * vs, gy0 + v[1] * vs, vs, vs, fill="none", stroke="currentColor",
                   stroke_width=2))
    for p in (dnode, snode):
        s.add(circle(p[0], p[1], 5, fill="currentColor", stroke="none"))
    sy = gy0 + sv[1] * vs + vs / 2
    dx_ = gx0 + dv[0] * vs + vs / 2
    dy_ = gy1 + 20
    s.add(path(f"M{f(gx0 + (sv[0] + 1) * vs)},{f(sy)} L{f(lxr - 6)},{f(sy)} "
               f"M{f(dx_)},{f(gy1)} L{f(dx_)},{f(dy_)} L{f(lxr - 6)},{f(dy_)}",
               stroke="currentColor", stroke_width=1.2, fill="none"))
    s.text_group()
    s.add(text(lxr, sy - 15, "a node on the", MAIN, weight=700))
    s.add(text(lxr, sy + 5, "free surface", MAIN, weight=700))
    s.add(text(lxr, sy + 27, "few or no faces of other", SEC))
    s.add(text(lxr, sy + 45, "cells in its voxel", SEC))
    s.add(text(lxr, dy_ + 5, "a node deep inside", MAIN, weight=700))
    s.add(text(lxr, dy_ + 26, "many faces of other cells", SEC))
    s.add(text(lxr, dy_ + 44, "in its voxel (drawn thick)", SEC))
    s.close()
    s.close()
    # voxel label on a plain corner square
    vy = gy0 + vs / 2
    s.add(leader(lxr - 6, vy, gx1 - vs / 2, vy))
    s.text_group()
    s.add(text(lxr, vy + 5, "one voxel", SEC, opacity=DIM))
    s.add(text(gx0, gy1 + 46, "2D slice: a face shows as a line segment", SEC, opacity=DIM))
    s.close()
    s.write(os.path.join(OUT, "fig-grid.svg"))
    return dict(cells=len(cells), deep_faces=len(segs), dv=dv, sv=sv)


# ---------------------------------------------------------------------------
# fig-threads
# ---------------------------------------------------------------------------
def fig_threads():
    costs = [2, 3, 2, 2, 3, 2,   6, 7, 5, 8, 6, 7,   3, 2, 4, 3, 2, 3,   2, 3, 2, 2, 3, 2]
    T, per = 4, 6
    heavy = [c >= 5 for c in costs]
    static = [[(i, costs[i]) for i in range(t * per, (t + 1) * per)] for t in range(T)]
    s_end = [sum(c for _, c in lane) for lane in static]
    s_max = max(s_end)
    dyn = [[] for _ in range(T)]
    heap = [(0, t) for t in range(T)]
    heapq.heapify(heap)
    for i, c in enumerate(costs):
        t0, t = heapq.heappop(heap)
        dyn[t].append((i, c, t0))
        heapq.heappush(heap, (t0 + c, t))
    d_end = [lane[-1][2] + lane[-1][1] for lane in dyn]

    lx, rx = 98, W - 14
    u = (rx - lx) / s_max
    lh, lgap = 22, 7
    s = Svg("fig-threads", W, 0, "Static blocks against dynamic scheduling",
            "Two timelines of the same loop over cells on four threads. With static blocks one "
            "thread gets all the expensive cells and runs long while the other three finish early "
            "and wait. With dynamic scheduling the expensive cells are spread over the threads and "
            "all four finish close together, sooner.")
    sq, sg = 19, 4
    yq = 36
    s.open(stroke="currentColor", stroke_width=1.1)
    for i in range(len(costs)):
        s.add(rect(lx + i * (sq + sg), yq, sq, sq, rx=2.5, fill="currentColor",
                   fill_opacity=0.26 if heavy[i] else 0.06))
    s.close()
    h0 = lx + 6 * (sq + sg) - 1
    h1 = lx + 12 * (sq + sg) - sg + 1
    s.add(path(f"M{f(h0)},{f(yq + sq + 4)} L{f(h0)},{f(yq + sq + 9)} L{f(h1)},{f(yq + sq + 9)} L{f(h1)},{f(yq + sq + 4)}",
               stroke="currentColor", stroke_width=1.0, fill="none", opacity=0.6))

    def timeline(y0, lanes_, ends, kind):
        for t in range(T):
            yy = y0 + t * (lh + lgap)
            s.open(stroke="currentColor", stroke_width=1.1)
            x = lx
            for item in lanes_[t]:
                i, c = item[0], item[1]
                if kind == "dynamic":
                    x = lx + item[2] * u
                s.add(rect(x + 0.8, yy, c * u - 1.6, lh, rx=2.5, fill="currentColor",
                           fill_opacity=0.26 if heavy[i] else 0.06))
                x += c * u
            s.close()
        xe = lx + max(ends) * u
        ybot = y0 + T * (lh + lgap) - lgap
        s.open_accent()
        for t in range(T):
            yy = y0 + t * (lh + lgap)
            xa = lx + ends[t] * u
            if xe - xa > 3:
                s.add(rect(xa + 0.8, yy, xe - xa - 0.8, lh, fill="currentColor", fill_opacity=0.08,
                           stroke="currentColor", stroke_width=1.0))
                s.add(path(hatch_d(xa + 0.8, yy, xe - xa - 0.8, lh, gap=7), stroke="currentColor",
                           stroke_width=1.0, fill="none"))
        s.close()
        s.add(line(xe + 0.5, y0 - 6, xe + 0.5, ybot + 6, stroke="currentColor", stroke_width=1.6, fill="none"))
        return xe, ybot

    ys = 130
    xe_s, yb_s = timeline(ys, static, s_end, "static")
    yd = yb_s + 76
    xe_d, yb_d = timeline(yd, dyn, d_end, "dynamic")
    s.add(line(xe_s + 0.5, yb_s + 28, xe_s + 0.5, yb_d + 6, stroke="currentColor", stroke_width=1.0,
               fill="none", stroke_dasharray="3 4", opacity=0.45))
    ya = yb_d + 34
    s.add(arrow(lx, ya, rx, ya, head=7, sw=1.1, opacity=0.6))
    H = ya + 26
    s.lines[0] = s.lines[0].replace(f'viewBox="0 0 {W} 0"', f'viewBox="0 0 {W} {f(H)}"')
    s.text_group()
    s.add(text(14, yq + 15, "the loop", MAIN, weight=600))
    s.add(text(lx, 24, "cells in index order", SEC, opacity=DIM))
    s.add(text((h0 + h1) / 2, yq + sq + 27, "expensive cells", SEC, anchor="middle", opacity=DIM))
    s.add(text(14, ys - 17, '<tspan font-weight="700">static:</tspan>'
               '<tspan dx="8">one contiguous block of cells per thread</tspan>', MAIN))
    s.add(text(14, yd - 17, '<tspan font-weight="700">dynamic:</tspan>'
               '<tspan dx="8">a free thread takes the next cell</tspan>', MAIN))
    for y0 in (ys, yd):
        for t in range(T):
            s.add(text(14, y0 + t * (lh + lgap) + 16, f"thread {t + 1}", SEC, opacity=DIM))
    s.add(text(xe_s - 8, yb_s + 23, "loop ends", SEC, anchor="end", opacity=DIM))
    s.add(text(xe_d + 9, yb_d + 23, "loop ends", SEC, anchor="start", opacity=DIM))
    s.add(text(rx, ya + 19, "time", SEC, anchor="end", opacity=DIM))
    s.close()
    s.open_accent()
    s.text_group()
    s.add(text(xe_s - 4, ys - 17, "hatched: waiting", MAIN, anchor="end", weight=600))
    s.close()
    s.close()
    s.write(os.path.join(OUT, "fig-threads.svg"))
    return dict(static_end=s_max, dynamic_ends=d_end, H=H)


# ---------------------------------------------------------------------------
# fig-scheduler
# ---------------------------------------------------------------------------
def fig_scheduler():
    H = 340
    s = Svg("fig-scheduler", W, H, "What the scheduler reaches in adaptive mode",
            "On the left, three groups of loops are each wired to a fixed schedule kind: mesh loops "
            "to static, contact for the node-node and node-face models to dynamic, polarization and "
            "internal forces to guided. On the right, the cell count and the divisions in the last "
            "50 iterations set a phase; the phase, the cost spread and the cell count then set a "
            "global schedule, which is drawn dashed and is wired to no loop.")
    bx, bw, kx, kw = 16, 196, 268, 104
    bh = 60
    rows = [(62, "mesh loops", None, "static"),
            (150, "contact", "(node-node, node-face)", "dynamic"),
            (238, "polarization +", "internal forces", "guided")]
    box = dict(stroke="currentColor", stroke_width=1.5, fill="currentColor", fill_opacity=0.06)
    for y, _, _, _ in rows:
        s.add(rect(bx, y, bw, bh, rx=8, **box))
        s.add(rect(kx, y + 9, kw, bh - 18, rx=8, stroke="currentColor", stroke_width=1.5, fill="none"))
        s.add(arrow(bx + bw + 3, y + bh / 2, kx - 4, y + bh / 2, head=8, sw=1.5))
    # right: two inputs set the phase; phase, cost spread and cell count set the global schedule
    cc = (430, 62, 104, 50)        # cell count
    dv = (548, 62, 158, 50)        # divisions in the last 50 iterations
    ph = (474, 156, 100, 42)       # phase
    cs = (592, 156, 114, 42)       # cost spread
    gl = (474, 244, 232, 48)       # global schedule
    for b in (cc, dv, ph, cs):
        s.add(rect(b[0], b[1], b[2], b[3], rx=8, **box))
    xa, xb = 504, 560              # cell count -> phase, divisions -> phase
    s.add(arrow(xa, cc[1] + cc[3] + 3, xa, ph[1] - 4, head=8, sw=1.5))
    s.add(arrow(xb, dv[1] + dv[3] + 3, xb, ph[1] - 4, head=8, sw=1.5))
    xp, xc = ph[0] + ph[2] / 2, cs[0] + cs[2] / 2
    s.add(arrow(xp, ph[1] + ph[3] + 3, xp, gl[1] - 4, head=8, sw=1.5))
    s.add(arrow(xc, cs[1] + cs[3] + 3, xc, gl[1] - 4, head=8, sw=1.5))
    # cell count also feeds the global schedule: down the left side, past the phase box
    xw, yw = 448, gl[1] + gl[3] / 2
    s.add(path(f"M{f(xw)},{f(cc[1] + cc[3] + 3)} L{f(xw)},{f(yw)}", stroke="currentColor",
               stroke_width=1.5, fill="none", stroke_linecap="round"))
    s.add(arrow(xw, yw, gl[0] - 4, yw, head=8, sw=1.5))
    s.open_accent()
    s.add(rect(gl[0], gl[1], gl[2], gl[3], rx=8, stroke="currentColor", stroke_width=2,
               fill="currentColor", fill_opacity=0.07, stroke_dasharray="7 5"))
    s.close()
    s.text_group()
    s.add(text(bx, 28, "per-loop schedule", 17, weight=700))
    s.add(text(bx, 48, "fixed when the solver starts", SEC, opacity=DIM))
    s.add(text(cc[0], 28, "detector", 17, weight=700))
    s.add(text(cc[0], 48, "runs every 50 iterations", SEC, opacity=DIM))
    for y, l1, l2, kind in rows:
        if l2 is None:
            s.add(text(bx + bw / 2, y + bh / 2 + 5.5, l1, MAIN, anchor="middle"))
        elif l2.startswith("("):
            s.add(text(bx + bw / 2, y + bh / 2 - 3, l1, MAIN, anchor="middle"))
            s.add(text(bx + bw / 2, y + bh / 2 + 16, l2, SEC, anchor="middle", opacity=DIM))
        else:
            s.add(text(bx + bw / 2, y + bh / 2 - 4, l1, MAIN, anchor="middle"))
            s.add(text(bx + bw / 2, y + bh / 2 + 16, l2, MAIN, anchor="middle"))
        s.add(text(kx + kw / 2, y + bh / 2 + 5.5, kind, MAIN, anchor="middle", family=MONO, weight=600))
    s.add(text(cc[0] + cc[2] / 2, cc[1] + cc[3] / 2 + 5, "cell count", 15, anchor="middle"))
    s.add(text(dv[0] + dv[2] / 2, dv[1] + 21, "divisions in the", SEC, anchor="middle"))
    s.add(text(dv[0] + dv[2] / 2, dv[1] + 39, "last 50 iterations", SEC, anchor="middle"))
    s.add(text(xp, ph[1] + ph[3] / 2 + 5.5, "phase", MAIN, anchor="middle"))
    s.add(text(xc, cs[1] + cs[3] / 2 + 5, "cost spread", 15, anchor="middle"))
    s.close()
    s.open_accent()
    s.text_group()
    s.add(text(gl[0] + gl[2] / 2, gl[1] + gl[3] / 2 + 5.5, "global schedule", MAIN, anchor="middle",
               weight=600))
    s.add(text(gl[0] + gl[2] / 2, gl[1] + gl[3] + 26, "no loop runs under it", MAIN, anchor="middle",
               weight=600))
    s.close()
    s.close()
    s.write(os.path.join(OUT, "fig-scheduler.svg"))


if __name__ == "__main__":
    fig_mesh()
    fig_forces()
    fig_divide()
    fig_iteration()
    print("grid:", fig_grid())
    print("threads:", fig_threads())
    fig_scheduler()
    print("wrote 7 figures to", OUT)
