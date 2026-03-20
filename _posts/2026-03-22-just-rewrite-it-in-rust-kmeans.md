---
layout: single
title: "Just rewrite it in Rust? A three-way k-means benchmark"
date: 2026-03-22T20:00:00+05:30
last_modified_at: 2026-03-22T20:00:00+05:30
categories: [blog]
tags: [rust, python, machine-learning, clustering, kmeans, benchmarking, performance, scikit-learn]
excerpt: "A hand-rolled Rust k-means is 5-27x faster and an order of magnitude leaner than pure Python — but the win has a shape."
math: true
header:
  overlay_image: /images/blog/bench-kmeans-rust/header.png
  overlay_filter: "rgba(15, 23, 34, 0.55)"
  teaser: /images/blog/bench-kmeans-rust/header.png
---

You profile a pipeline, the k-means call lights up red, and your hand is already reaching for Rust. It's a reflex — the loop is hot, Rust is fast, rewrite it. A single k-means fit at n = 1,000 takes pure-Python NumPy **0.86 s** and scikit-learn **3.21 s**; my hand-rolled Rust binary does it in **0.05 s**. That's a ~17x edge over NumPy and up to ~65x over the industrial BLAS-backed library scikit-learn at small N (n=1k: 3.21 s vs 0.05 s) — though that lead is only 1.5x on the mean, and it reverses at the largest size I tested, n=128k, where sklearn's BLAS vectorization finally overtakes. The instinct looks dead right — until you measure the second data point.

Because the win has a *shape*, and the shape is the interesting part. I built three implementations of Lloyd's algorithm — pure-Python/NumPy, hand-rolled Rust (serial + Rayon-parallel), and scikit-learn as the industrial baseline — and swept all three implementations across **8 dataset sizes × 7 feature counts × 6 cluster counts** — 336 configurations each, 1,008 runs in total — on an **Apple M4 Max**. All the code is in the [repo](https://github.com/nilesh-patil/bench-kmeans-rust), and there's a [live companion](https://nilesh-patil.github.io/bench-kmeans-rust/) with a WASM demo and the full Plotly dashboards if you want to poke at the raw curves. This post settles whether the rewrite instinct pays off — measured, not asserted.

> **TL;DR.** Hand-rolled Rust is median **7.32x** faster than pure-Python NumPy (median 0.21 s vs 1.56 s). The win has a shape: **17.4x @ n=1k** decaying to **4.83x @ n=128k**, where sklearn's BLAS finally overtakes. Memory is the durable win: Rust's peak RSS is the leanest at every size — averaging **~11–12x** less than both across the sweep. Rayon parallelism buys only **~1.09–1.16x** over serial Rust and only at scale — below n=8k it's ~1.8x *slower*. On accuracy, single-seed Rust/Python land ARI **0.6624 / 0.7434** against sklearn's perfect **1.0000** — an init-strategy gap (`n_init=10`), not an algorithm gap, and the cheapest to close: k-means++ alone cuts inertia **37–54%**.

---

## The algorithm is identical in all three — only the engine changes

Two steps, alternating.

1. **Assignment.** Hand each point to its closest centroid.
2. **Update.** Move each centroid to the mean of the points that chose it.

Repeat until the labels stop moving. In symbols:

$$c_i = \arg\min_j \lVert x_i - \mu_j \rVert^2 \qquad\qquad \mu_j = \frac{1}{\lvert C_j \rvert}\sum_{x_i \in C_j} x_i$$

Read left-to-right: assign point $x_i$ to the nearest center $\mu_j$ (that assignment is $c_i$); then set each center to the average of its members ($\lvert C_j \rvert$ is how many points chose cluster $j$).

Both steps are quietly minimizing the same thing — the total squared distance from each point to its cluster's center (the within-cluster sum of squares). Assigning to the nearest centroid can't increase it; recomputing the mean can't either. That's why it converges: it's coordinate descent, the same alternating-minimization shape as an EM loop. If you've trained anything, you already own the mental model.

So the field is level — same math in Python, in Rust, and inside sklearn. Every difference I'm about to show is attributable to two things and two things only: the *engine* that runs the arithmetic, and the *initialization strategy*. Nothing is a smarter algorithm.

The engines differ in how they touch memory. Pure-Python NumPy loops over the k centroids in Python, computing one column of the distance matrix per iteration — `distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)` — rather than broadcasting a single pairwise-distance tensor across all centroids at once; Rust runs explicit loops over contiguous slices with no interpreter and no per-operation allocation; sklearn hands the same distance computation to a BLAS GEMM — decades of tuned linear algebra. Same arithmetic, three radically different memory-and-dispatch profiles.

The one asymmetry that matters for accuracy: both hand-rolled implementations default to `random` initialization (k-means++ is opt-in); scikit-learn keeps its own default — k-means++ with **`n_init=10`** restarts — which is exactly the init-strategy gap behind its higher ARI. My Python and Rust use a single seed. I'm flagging it now so the quality result later reads as earned, not as a Rust failure; the full mechanism waits for the accuracy section. Same dataset throughout: synthetic Gaussian blobs, seed 42.

![A single-seed k-means run stalling in a poor local optimum](/images/blog/bench-kmeans-rust/convergence_random.gif)
*Random init can drop Lloyd's into a poor local optimum and leave it stuck — the centroids settle on a split that lowers the objective locally but never recovers the true clusters. Running several random restarts and keeping the best is the standard escape.*

---

## The speed win is real — and biggest exactly where you'd reach for it

Across the full sweep, hand-rolled Rust is **median 7.323x** faster than pure-Python NumPy — median 0.21 s versus 1.56 s, with sklearn at 3.73 s. I lead with the median on purpose. The *mean* speedup is lower, **5.169x**, and the gap between those two numbers is itself a lesson.

If you report p50 latency instead of the average because one slow request skews everything, you already own this. The slowest single run in the sweep hit **2,351 s** (Python, n=128k, f=128, k=64 — **64.7x** the Python mean), and a handful of other huge-N configs sit right beside it. Those huge-N runs are exactly where Rust's lead has already shrunk to ~5x. Because they dwarf every other config in absolute seconds, a ratio-of-totals (the mean) is effectively a weighted average that puts almost all its weight on those low-advantage runs — so it reports 5.17x. The median ignores the weighting and reports the typical config: 7.32x.

The win concentrates at small and medium N — exactly the regime where you profile a notebook and decide a rewrite is worth it. At n=1k, Rust finishes in **0.049 s** against Python's **0.861 s**: a **17.4x** speedup, and sub-100ms where Python is already laggy.

![Runtime vs dataset size on log-log axes for Python, sklearn, and Rust](/images/blog/bench-kmeans-rust/figure-1-runtime-vs-size.svg)
*Runtime vs N (log-log), three implementations. On log-log axes a power law plots as a straight line whose slope IS the growth exponent, so the steepest line wins the high-N race: Rust's **1.057** (super-linear) against sklearn's near-flat **0.286** and Python's **0.709**. Rust (0.049 → 0.484 → 1.34 → 4.00 → 11.4 → 38.64 s at n = 1k/8k/16k/32k/64k/128k) sits lowest at every size except the rightmost, where its steeper slope finally lifts it above sklearn.*

Read the slopes, not the heights: the flex is real, but those diverging slopes are foreshadowing its own ceiling.

---

## The first ceiling: BLAS overtakes at scale

A steeper line always eventually overtakes a flatter one — that's what different slopes mean — so Rust's **1.057** was always going to cross sklearn's **0.286**. The only question was where.

![Speedup over pure-Python vs dataset size, for Rust and sklearn](/images/blog/bench-kmeans-rust/figure-2-speedup-curve.svg)
*Speedup over Python vs N. The Rust curve decays (17.4x @1k → 4.83x @128k) while the sklearn curve climbs (0.27x @1k → 6.21x @128k), crossing parity-with-Python between n=16k and n=32k (already 1.67x at 32k) and crossing above Rust by n=128k. sklearn at 0.27x means it's nearly 4x slower than NumPy at small N — n_init=10 plus startup overhead crushes it there. Two curves, opposite directions.*

The Rust speedup-over-Python **decays from 17.4x at n=1k to 4.83x at n=128k**. The sklearn speedup *climbs* from **0.27x at n=1k** — yes, nearly 4x slower than plain NumPy at small N, because `n_init=10` plus library startup overhead is brutal when there's almost no data — to **6.21x at n=128k**, crossing parity with Python between n=16k and n=32k and crossing above Rust by the end.

The crossover, plainly: at n=128k, Rust's mean is **38.64 s** and sklearn's is **30.06 s**. sklearn wins. The cause is exactly the exponents — Rust's **1.057** (my naive O(N·k·d) loops, with cache pressure nudging it just past linear) against sklearn's **0.286** (BLAS GEMM keeps the distance matrix in cache and saturates SIMD and memory bandwidth). This is mechanism, not magic: sklearn is *not* a smarter algorithm. It's the same Lloyd's iteration handed to decades of tuned linear algebra.

And the thesis edge holds even here. At the crossover, Rust is **still 4.83x faster than pure Python**. The rewrite never loses to NumPy — it loses to BLAS. The rule is: rewrite in Rust to beat your NumPy loop; reach for sklearn (or any BLAS-backed code) once N gets very large. A BLAS-backed Rust — `ndarray` with a GEMM-based distance — would almost certainly reclaim the high-N regime, which is the tell: the lesson isn't "Rust loses," it's "naive loops lose to vectorization once N is big enough to amortize vectorization's fixed overhead." It's the same lesson as "don't hand-write a Python for-loop when `np.dot` exists." My loops were the ceiling, not Rust.

---

## The parallel reality: Rayon is not the 14x you hoped for

The most seductive over-claim about a Rust rewrite is "add threads, get 14x." So I added threads — Rayon, the data-parallelism library — and measured it properly on a fresh M4 Max re-run. The answer is modest, and the *why* is more instructive than the number.

Peak speedup over serial Rust climbs with N but plateaus near **~1.15x** — nowhere near the 14-core ideal.

![Rayon speedup over serial Rust vs thread count, one curve per dataset size](/images/blog/bench-kmeans-rust/figure-4-parallel-scaling.svg)
*Rayon speedup vs threads, fresh M4 Max medians-of-3 (f=16, k=16, k_max=8). Peaks of 1.087x (n=16k), 1.118x (n=50k), 1.155x (n=128k) — the curves barely lift off 1.0, nowhere near the 14x ideal. The two smaller datasets peak around 4 threads and roll over by 14, while at n=128k the curve climbs monotonically to its peak at all 14 threads — because each timed fit is too small a granule to amortize thread spawn/join cost.*

That roll-over is the granularity story in full. The benchmark fits k = 1 through 8 for every configuration, so most of the timed work is tiny small-k fits where Rayon's thread-spawn and join setup dominates the actual arithmetic — the absolute times are sub-second even at 128k. There simply isn't enough work per task to amortize the coordination. At n ≤ 8k, parallel is about **1.8x slower** than serial (the measured ratio is 0.554x) — spawning and joining threads costs more than the tiny per-fit arithmetic they're splitting.

Here's the Python-engineer bridge: a single-threaded NumPy loop only touches one of these 14 cores — but contrary to the usual reflex, the GIL isn't really the culprit. NumPy drops the GIL the moment it enters its C loops, which is exactly why a BLAS-backed call CAN spread across cores. The reason our pure-Python baseline stays pinned to one core is simpler: it never asks for more than one. The Lloyd iterations run as a serial Python loop, each NumPy call executes single-threaded, and nothing in the implementation spawns parallel work. (The GIL is the real wall the moment you reach for Python's own `threading` on CPU-bound pure-Python code — just not here.) Rust, by contrast, has no such ceiling: Rayon can fan the same work across all 14 cores at will. But threads only pay when each task is big enough to dwarf the spawn cost, the same reason DataLoader workers cost more than they save when each one does almost nothing. At this workload's granularity (sub-second k=1..8 fits), Rayon is a rounding error on top of the serial win — and that's fine, because the serial win was already the headline. These fresh M4 Max medians-of-3 directly measure n=16k, 50k, and 128k. The honest boundary isn't the dataset sizes I covered; it's granularity: each timed fit is k=1..8, so even at 128k the per-task work stays small.

---

## The memory win is the one that holds at every size

If the speed advantage has a ceiling, the memory advantage has none. Rust uses an order of magnitude less peak memory than either competitor, at *every* size, with no crossover anywhere — the cleanest result in the whole project.

The defensible headline is absolute mean peak RSS: sklearn uses **12.03x** more memory than Rust, and Python **10.96x** more. In raw numbers that's Rust at **23.33 MB** mean against Python's **255.63 MB** and sklearn's **280.74 MB**. "An order of magnitude leaner" is literally true.

![Left: memory per 1k samples by implementation. Right: peak memory vs dataset size](/images/blog/bench-kmeans-rust/figure-3-memory.svg)
*Left: MB per 1k samples — Rust 0.83, Python 25.18, sklearn 47.88 (a ~30x and ~58x spread, but inflated at small N by the fixed ~20–50 MB interpreter base and ~5 MB Rust binary — read this bar as illustrative). Right: peak RSS vs N — Rust stays nearly flat versus the other two (0.03–234 MB across all configs, vs Python 67.92–2,361 and sklearn 161.62–1,075). Rust stays lean because its assignment step never builds a distance matrix at all — it walks each point against every centroid in a tight nested loop and keeps only a running nearest-centroid index, so peak memory tracks the dataset itself. Pure-Python NumPy balloons because its vectorized distance step allocates a fresh N-by-k matrix on every iteration. sklearn's larger footprint is a different story: its compiled chunked kernels don't materialize that matrix either — its overhead comes from running n_init=10 restarts and the interpreter base. The number to take to the bank is the absolute ~11–12x; the per-1k ratio is the inflated one.*

Those two ratios disagree, and the smaller one is the honest one. The per-1k-sample bar shows a ~30x and ~58x spread, which looks more impressive than ~12x — but it's inflated at small N by fixed overhead that has nothing to do with the algorithm: the Python interpreter carries ~20–50 MB of base footprint and the Rust binary ~5 MB, and at n=1,000 that base dominates the measurement. So the per-1k bar is illustrative and the **~11–12x absolute** is the number I'll defend.

The mechanism is one an ML reader already owns. NumPy's vectorized distance step materializes the full N-points-by-k-centroids distance matrix in float64 *every iteration* — vectorization in Python is, concretely, "allocate a big array," and that array *is* the memory cost. Rust's per-sample memory rate is tiny — about 0.83 MB per 1,000 points versus 25.18 (Python) and 47.88 (sklearn) — so on absolute peak RSS it uses roughly 11–12x less memory than either Python or scikit-learn at every size. Its peak footprint still grows with N (from a few hundred KB at n=1k up to ~234 MB at the largest configs), but it grows from a far lower base and at a far gentler slope, because the data sits in a compact `Vec<DataPoint>` of f64s with no interpreter or BLAS scratch overhead riding along. That's the right panel of the figure: the Rust line stays low and climbs gently while the other two climb steeply. Crucially, this advantage does *not* decay with N the way the speed advantage does — it's the most durable part of the rewrite, and it's why Rust wins the memory axis of the quality trade-off next.

---

## The second cost: single-seed init costs you accuracy

The last twist is on the quality axis. My Rust implementation produces the *worst* clusterings of the three. The cause is not the algorithm — it's the init strategy.

Caveat first — the external-quality grid is narrow: **k=8 fixed, n ≤ 8,000, f ≤ 8, 12 runs**. On that grid, mean Adjusted Rand Index against ground truth is sklearn **1.0000** (perfect on all **12/12** runs), Python **0.7434**, and Rust **0.6624** (rust_parallel is bit-identical). Worst-case ARI is Python **0.3831** and Rust **0.3526**. The internal metrics tell the same story across the *full* sweep: silhouette (how cleanly separated the clusters are, higher better) is sklearn **0.93** / Python **0.64** / Rust **0.62**, and Davies-Bouldin (cluster overlap, lower better) is sklearn **0.09** against Python **1.83** and Rust **1.96** — a ~20x gap, the sharpest single-number quality contrast in the whole dataset.

But this is an init-*strategy* gap, not an algorithm-quality gap. sklearn's perfection comes entirely from `n_init=10`: ten random starts, keep the best inertia. My single-seed Python and Rust occasionally land in a poor local optimum and stay there — exactly the stall the first GIF showed. It's the difference between training once and training with ten seeds and keeping the best checkpoint. We already glimpsed the bill for those ten restarts back in the speed section — `n_init=10` is a big part of why sklearn runs at 0.27x of NumPy at small N. The accuracy isn't free; you pay for it in repeated fits.

The principled fix is k-means++, and the intuition comes first. You want the initial centers spread out, not clumped — so pick each new seed preferentially from the regions farthest from the seeds you already have. It's the same instinct as choosing diverse few-shot examples instead of ten near-duplicates. The formalization of that instinct is to sample each next seed with probability proportional to its squared distance from the nearest existing center:

$$p(x) \propto D(x)^2$$

That single rule spreads seeds apart, so even one start tends to land near a good basin. In a pure-Python ablation (10 seeds per config), k-means++ cut final inertia by **37–54%** versus random — and the advantage *grows* with k and dimensionality: **37.2%** at (n1000, d2, k4), **53.1%** at (n8000, d8, k16), **54.4%** at (n32000, d16, k32).

![k-means++ initialization spreading seeds and converging quickly](/images/blog/bench-kmeans-rust/convergence_kpp.gif)
*k-means++ spreads seeds via D²-weighted sampling — each new center lands far from the existing ones, so the run drops into the right basins and converges in a handful of iterations. This is the 37–54% inertia reduction over random init, visualized.*

Step back and all three implementations land on a clean Pareto frontier across accuracy, speed, and memory:

![ARI vs runtime with bubble size encoding peak memory](/images/blog/bench-kmeans-rust/figure-5-quality-pareto.svg)
*ARI vs runtime on the narrow quality grid (k=8, n≤8k); bubble area = peak memory. Rust: 0.058 s / 3.86 MB / ARI 0.6624. sklearn: 5.26 s / 179 MB / ARI 1.0000. Python: 1.59 s / 81.6 MB / ARI 0.7434. Rust owns the fast-and-lean bottom-left; sklearn buys perfect accuracy with the most time and memory; no implementation is Pareto-dominant.*

Read figure-5 as a Pareto picture: Rust owns the fast-and-lean corner with the lowest accuracy; sklearn owns the perfect-accuracy corner at the highest cost; Python sits between. No one dominates — you choose your corner. And the corner Rust is missing is *cheap* to reach: bolting on restarts or k-means++ closes the one axis where it loses without touching speed or memory. That's a fundamentally different kind of fix than the BLAS crossover, which needs real vectorization work.

---

## When the rewrite pays off

One beat before the verdict, because none of this is about speed: k-means itself has assumptions, and no language fixes them. It carves space by nearest-centroid — Voronoi cells — so it implicitly assumes convex, roughly round, similarly-sized clusters.

![k-means slicing concentric rings into pie wedges](/images/blog/bench-kmeans-rust/convergence_circles.gif)
*Two concentric rings — k-means pie-slices them into wedges instead of separating inner from outer, because Voronoi cells can't wrap around. This failure is the model's assumption, shared identically by all three implementations; no rewrite touches it.*

That's the one limit no implementation choice can buy back; everything below is a choice you actually get to make. So, three lessons, as a decision rule.

**Lesson 1 — the rewrite wins exactly where you reach for it.** If your bottleneck is a NumPy k-means at small or medium N, yes: hand-rolled Rust is median **7.3x** faster and **~11–12x** leaner, and the memory win never erodes with scale. The "just rewrite it" instinct is vindicated in its home regime.

**Lesson 2 — vectorization beats naive native code at scale.** The speed win *decays* with N (**17.4x → 4.83x**), and sklearn's BLAS overtakes my raw Rust loops at n=128k, having crossed parity-over-Python near **n=32k**. Rewrite to beat your interpreter, not to beat tuned linear algebra — a BLAS-backed `ndarray` Rust is the actionable next step for the high-N regime.

**Lesson 3 — the accuracy gap was an init bug, not an algorithm gap.** sklearn's perfect ARI came from `n_init=10`, full stop. The cheapest "rewrite" in the whole project is adding restarts or k-means++ (**37–54%** lower inertia) to the Rust implementation — closing the one axis it lost on without spending a single point of speed or memory. Spend your effort on init, not threads.

Caveats, briefly: single M4 Max, narrow ARI grid (k=8, n ≤ 8k), parallel bounded by per-fit granularity (k=1..8), heavy-tailed runtimes — trust the median. The win is real and it has a shape: huge and lean at small/medium N, modest from parallelism, and bounded above by BLAS and by `n_init`.

- **Code**: [github.com/nilesh-patil/bench-kmeans-rust](https://github.com/nilesh-patil/bench-kmeans-rust) — all three implementations plus the sweep harness.
- **Live demo + dashboards**: [nilesh-patil.github.io/bench-kmeans-rust](https://nilesh-patil.github.io/bench-kmeans-rust/) — the interactive WASM demo and the full Plotly curves.
- **Project page**: [Python vs Rust: k-means at scale](/portfolio/bench-kmeans-rust/) — the side-projects entry.

Clone it, run the sweep on your own hardware, and watch where *your* crossover lands — the exponents are a property of your CPU's cache and BLAS as much as the algorithm.
