"""A deformed icosphere projected to 2D, for drawing a cell as a triangle mesh."""
import math
import numpy as np

from kit import f, path, circle, poly_d


def icosphere(sub=1):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
         (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(sub):
        cache, F2 = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[k] = len(V) - 1
            return cache[k]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return np.array(V), F


def rot_matrix(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Cell:
    """radii: (a, b, c) along body x, y, z. clamp: list of (sign, value): body-x coordinates
    beyond value (in the direction of sign) are flattened onto the plane x = value.
    theta: in-plane rotation of the body x axis. Viewer looks along z."""

    def __init__(self, cx, cy, radii, seed, clamp=(), theta=0.0, sub=1, jitter=0.10, rnoise=0.035):
        rng = np.random.default_rng(seed)
        V, F = icosphere(sub)
        V = V + jitter * rng.normal(size=V.shape) * 0.5
        V /= np.linalg.norm(V, axis=1)[:, None]
        V = V @ rot_matrix(rng).T
        V = V * (1 + rnoise * rng.normal(size=(len(V), 1)))
        V = V * np.array(radii)[None, :]
        for sign, val in clamp:
            if sign > 0:
                V[:, 0] = np.minimum(V[:, 0], val)
            else:
                V[:, 0] = np.maximum(V[:, 0], val)
        c, s = math.cos(theta), math.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        V = V @ Rz.T
        self.V3 = V
        self.P = V[:, :2] + np.array([cx, cy])[None, :]
        self.F = F
        ctr = V.mean(axis=0)
        self.front, self.shade = [], []
        L = np.array([-0.45, -0.62, 0.64])
        L /= np.linalg.norm(L)
        for a, b, c_ in F:
            n = np.cross(V[b] - V[a], V[c_] - V[a])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                self.front.append(False)
                self.shade.append(0.0)
                continue
            n /= nn
            if np.dot(n, (V[a] + V[b] + V[c_]) / 3 - ctr) < 0:
                n = -n
            self.front.append(n[2] > 0.02)
            self.shade.append(float(max(0.0, np.dot(n, L))))
        self.edges = {}
        for fi, (a, b, c_) in enumerate(F):
            for u, v in ((a, b), (b, c_), (c_, a)):
                self.edges.setdefault((min(u, v), max(u, v)), []).append(fi)
        self.front_nodes = sorted({i for fi, fc in enumerate(F) if self.front[fi] for i in fc})

    def edge_sets(self):
        inner, rim, back = [], [], []
        for e, fs in self.edges.items():
            k = sum(1 for fi in fs if self.front[fi])
            (inner if k == len(fs) else rim if k > 0 else back).append(e)
        return inner, rim, back

    def seg_d(self, edges):
        return " ".join(f"M{f(self.P[a][0])},{f(self.P[a][1])} L{f(self.P[b][0])},{f(self.P[b][1])}"
                        for a, b in sorted(edges))

    def draw(self, shade=True, back=True, node_r=2.6, sw_in=1.1, sw_rim=1.6, lo=0.03, hi=0.12,
             skip_faces=()):
        out = []
        if shade:
            # a handful of opacity steps keeps the markup small
            steps = 5
            buckets = {}
            for fi, fc in enumerate(self.F):
                if not self.front[fi] or fi in skip_faces:
                    continue
                k = min(steps - 1, int((1 - self.shade[fi]) * steps))
                buckets.setdefault(k, []).append(poly_d([tuple(self.P[i]) for i in fc]))
            for k in sorted(buckets):
                op = lo + (hi - lo) * k / (steps - 1)
                out.append(path(" ".join(buckets[k]), fill="currentColor", fill_opacity=round(op, 3),
                                stroke="none"))
        inner, rim, bk = self.edge_sets()
        if back and bk:
            out.append(path(self.seg_d(bk), stroke="currentColor", stroke_width=0.8, fill="none",
                            opacity=0.2, stroke_linecap="round"))
        out.append(path(self.seg_d(inner), stroke="currentColor", stroke_width=sw_in, fill="none",
                        opacity=0.85, stroke_linecap="round"))
        out.append(path(self.seg_d(rim), stroke="currentColor", stroke_width=sw_rim, fill="none",
                        stroke_linecap="round"))
        if node_r:
            out.append("<g fill=\"currentColor\" stroke=\"none\">")
            for i in self.front_nodes:
                out.append(circle(self.P[i][0], self.P[i][1], node_r))
            out.append("</g>")
        return out

    def nearest_node(self, x, y):
        return min(self.front_nodes, key=lambda i: (self.P[i][0] - x) ** 2 + (self.P[i][1] - y) ** 2)

    def nearest_inner_edge(self, x, y):
        inner, _, _ = self.edge_sets()
        return min(inner, key=lambda e: ((self.P[e[0]][0] + self.P[e[1]][0]) / 2 - x) ** 2
                   + ((self.P[e[0]][1] + self.P[e[1]][1]) / 2 - y) ** 2)

    def nearest_face(self, x, y):
        best, bd = None, 1e18
        for fi, fc in enumerate(self.F):
            if not self.front[fi]:
                continue
            c = self.P[list(fc)].mean(axis=0)
            d = (c[0] - x) ** 2 + (c[1] - y) ** 2
            if d < bd:
                best, bd = fi, d
        return best
