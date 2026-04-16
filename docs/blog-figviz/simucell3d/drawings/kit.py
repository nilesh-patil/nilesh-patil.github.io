"""Small SVG helpers for the seven line drawings. Output rules: one element per
line, explicit closing tags, currentColor only, no blank lines."""
import math
import random

FONT = "-apple-system, system-ui, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
ACCENT = 'style="color:var(--global-accent, currentColor)"'


def f(x):
    s = f"{x:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    if s == "-0":
        s = "0"
    return s


def attrs(**kw):
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        k = k.rstrip("_").replace("_", "-")
        if isinstance(v, float):
            v = f"{v:.3g}" if abs(v) < 1 else f(v)
        out.append(f'{k}="{v}"')
    return (" " + " ".join(out)) if out else ""


class Svg:
    def __init__(self, fid, w, h, title, desc):
        self.fid, self.w, self.h = fid, w, h
        self.lines = [
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-labelledby="{fid}-t {fid}-d" '
            f'style="width:100%;height:auto;max-width:{w}px;color:inherit" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<title id="{fid}-t">{title}</title>',
            f'<desc id="{fid}-d">{desc}</desc>',
        ]

    def add(self, s):
        if isinstance(s, (list, tuple)):
            for x in s:
                self.add(x)
        elif s:
            self.lines.append(s)

    def open(self, **kw):
        self.lines.append(f"<g{attrs(**kw)}>")

    def open_accent(self, **kw):
        self.lines.append(f"<g {ACCENT}{attrs(**kw)}>")

    def close(self):
        self.lines.append("</g>")

    def text_group(self, **kw):
        kw.setdefault("fill", "currentColor")
        kw.setdefault("stroke", "none")
        self.lines.append(f'<g font-family="{FONT}" aria-hidden="true"{attrs(**kw)}>')

    def write(self, path):
        self.lines.append("</svg>")
        body = "\n".join(l for l in self.lines if l.strip())
        with open(path, "w") as fh:
            fh.write(body + "\n")


def line(x1, y1, x2, y2, **kw):
    return f'<path d="M{f(x1)},{f(y1)} L{f(x2)},{f(y2)}"{attrs(**kw)}></path>'


def path(d, **kw):
    return f'<path d="{d}"{attrs(**kw)}></path>'


def poly_d(pts, close=True):
    d = "M" + " L".join(f"{f(x)},{f(y)}" for x, y in pts)
    return d + (" z" if close else "")


def rect(x, y, w, h, rx=None, **kw):
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"{attrs(rx=rx, **kw)}></rect>'


def circle(x, y, r, **kw):
    return f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r)}"{attrs(**kw)}></circle>'


def text(x, y, s, size=14, anchor=None, weight=None, opacity=None, italic=False, family=None, **kw):
    a = attrs(font_size=size, text_anchor=anchor, font_weight=weight, opacity=opacity,
              font_style="italic" if italic else None, font_family=family, **kw)
    return f'<text x="{f(x)}" y="{f(y)}"{a}>{s}</text>'


def sub(base, s, size=10):
    """base with a subscript, as tspans (baseline-shift is not portable)."""
    return (f'{base}<tspan dy="3.5" font-size="{size}" font-style="normal">{s}</tspan>'
            f'<tspan dy="-3.5" font-size="{size}"> </tspan>')


def arrow(x1, y1, x2, y2, head=6.5, hw=None, sw=1.3, **kw):
    """Line with an explicit filled head at (x2, y2). Returns [line, head]."""
    hw = hw if hw is not None else head * 0.42
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    px, py = -uy, ux
    out = []
    if L > head:
        out.append(line(x1, y1, bx + ux * 0.5, by + uy * 0.5, stroke="currentColor",
                        stroke_width=sw, fill="none", **kw))
    out.append(path(poly_d([(x2, y2), (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)]),
                    fill="currentColor", stroke="none", **{k: v for k, v in kw.items() if k == "opacity"}))
    return out


def hatch_d(x, y, w, h, gap=6.0):
    """Diagonal hatch lines (45 deg) clipped to a rectangle, as one path d."""
    segs = []
    c = -h + gap / 2
    while c < w:
        # line y' = x' - c in rect-local coords, x' in [0,w], y' in [0,h]
        xa, xb = max(0.0, c), min(w, c + h)
        if xb > xa:
            segs.append(f"M{f(x + xa)},{f(y + h - (xa - c))} L{f(x + xb)},{f(y + h - (xb - c))}")
        c += gap
    return " ".join(segs)


def blob(cx, cy, r, n, seed, jit=0.07, rot=0.0, sx=1.0, sy=1.0, ajit=0.10):
    rng = random.Random(seed)
    pts = []
    for i in range(n):
        a = rot + 2 * math.pi * (i + rng.uniform(-ajit, ajit)) / n
        rr = r * (1 + rng.uniform(-jit, jit))
        pts.append((cx + sx * rr * math.cos(a), cy + sy * rr * math.sin(a)))
    return pts


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def out_normal(p, q, c):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = dy / L, -dx / L
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    if (mx - c[0]) * nx + (my - c[1]) * ny < 0:
        nx, ny = -nx, -ny
    return nx, ny


def scale_pts(pts, c, s):
    return [(c[0] + (x - c[0]) * s, c[1] + (y - c[1]) * s) for x, y in pts]


def dots(pts, r=2.6, **kw):
    return [circle(x, y, r, **kw) for x, y in pts]


def clip_convex(subject, clipper):
    """Sutherland-Hodgman: subject polygon clipped by a convex clipper polygon."""
    def inside(p, a, b, sign):
        return sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0

    def inter(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    area = sum(clipper[i][0] * clipper[(i + 1) % len(clipper)][1]
               - clipper[(i + 1) % len(clipper)][0] * clipper[i][1] for i in range(len(clipper)))
    sign = 1 if area > 0 else -1
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b, sign):
                if not inside(s, a, b, sign):
                    out.append(inter(s, e, a, b))
                out.append(e)
            elif inside(s, a, b, sign):
                out.append(inter(s, e, a, b))
            s = e
    return out
