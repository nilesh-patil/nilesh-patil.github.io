---
layout: single
title: "Where the 15x went: benchmarking a parallel Rust k-means rewrite"
date: 2026-03-22T20:00:00+05:30
last_modified_at: 2026-06-13T20:00:00+05:30
categories: [blog]
tags: [rust, python, machine-learning, clustering, performance, parallelism]
excerpt: "Four k-means engines on one grid  pure-Python NumPy, serial Rust, a Rayon-parallel Rust, and scikit-learn. The hand-rolled Rust wins 15x at a thousand points, 3.5x at a quarter-million, and loses one corner outright. A field report on where the speedup went, and what parallelism and a matched init actually buy."
math: true
header:
  image: /images/blog/bench-kmeans-rust/cover.jpg
  teaser: /images/blog/bench-kmeans-rust/cover.jpg
redirect_from:
  - /posts/just-rewrite-it-in-rust-kmeans/
---

Open the [live demo](https://www.nilesh42.science/bench-kmeans-rust/demo/), pick a point distribution, and hit run. The first pass clusters your points with the hand-rolled Rust k-means from this post, compiled to WebAssembly. The second pass is a pure-JavaScript k-means on the *same* data and the *same* seed, timed beside it with `performance.now()` on your machine. That gap is what this whole post measures.

I tried the obvious thing, then measured it properly. At n=1,000, a full elbow sweep (load the CSV, fit every k from 1 to K, write results) takes pure-Python NumPy 0.43 s and scikit-learn 1.57 s. My hand-rolled Rust binary does it in 0.029 s, roughly 15x over NumPy and 54x over the industrial, BLAS-backed scikit-learn. That 15x is the small-n, single-run figure; the paired median across the n=1k cells lands at 14.5x (see the speedup curve below). It makes the rewrite look like a no-brainer. Then I gave everyone the same initialization, swept out to a quarter-million points, and watched the 15x shrink to 3.5x, with one corner of the grid flipping outright.

So instead of trusting one number I ran a contest. **K-means, four ways**, all running the *same* Lloyd iteration, on the same grid, on one machine:

1. **Pure-Python NumPy**  about 150 lines, the baseline most pipelines start from.
2. **Hand-rolled serial Rust**  explicit loops, no interpreter, no per-operation allocations.
3. **Rayon-parallel Rust**  the *same* Rust binary, run with `--parallel --threads 0`.
4. **scikit-learn** at `n_init=1`  the industrial standard, given a single start like everyone else.

The parallel build is a full contestant here. It runs on the same grid and reports the same metrics, so its thread bonus shows up as a measured number.

<div class="notice--info" markdown="1">
**Four ways at a glance** (648 runs, Apple M4 Max, all implementations on identical k-means++ init):

<div style="overflow-x:auto" markdown="1">

| Implementation | Median wall time | Speedup vs Python | Median RSS / 1k rows | Median ARI |
|---|---|---|---|---|
| **Pure-Python NumPy** | 0.806 s | 1.0x (baseline) | 7.41 MB | 1.00 |
| **Serial Rust** | 0.201 s | **4.5x** | **0.61 MB** | 1.00 |
| **Parallel Rust (Rayon)** | 0.197 s | **5.1x** | 0.73 MB | 1.00 |
| **scikit-learn** (`n_init=1`) | 1.843 s | 0.44x | 12.63 MB | 1.00 |

</div>

Speed decays with size (15x to 3.5x for serial Rust); Rayon tops out at 1.32x over serial; the memory ranking never flips; matched init ties the quality.

*Speedup column, definitions made explicit.* The two Rust figures are medians of paired per-dataset ratios (Python time over Rust time, one ratio per dataset). scikit-learn's 0.44x is instead the ratio of overall median wall times (0.806 / 1.843), because its standing against Python reverses with size  0.27x at n=1k, 1.80x at n=256k  so no single paired median describes it honestly.
</div>

## The setup

One end-to-end CLI run per configuration: load CSV, fit k = 1..K, write CSV, exit. I time the whole subprocess instead of an in-process fit, because the subprocess is what a pipeline actually pays for. The grid is 9 sizes (1k to 256k, doubling) × 3 feature counts (2/8/32) × 2 cluster budgets (k_max 8 or 32) × 3 repeats, on synthetic Gaussian blobs, giving 162 dataset cells × 4 implementations = 648 runs. One fairness control carries the most weight. Every implementation uses k-means++ initialization, and scikit-learn runs at `n_init=1`, a single start like everyone else. This is a rerun of an earlier experiment that didn't control init, and that control changes one headline result, which I'll get to.

Metrics are subprocess wall time, sampled peak RSS (polled every 10 ms, so an estimate that can undercount brief spikes rather than a kernel max), and ARI/NMI against the ground-truth blob labels.

The four ways run identical math with radically different execution profiles. The pure-Python NumPy version loops over centroids, filling one column of the distance matrix at a time with `np.linalg.norm(X - centroids[k], axis=1)`. Serial Rust runs explicit loops over its own data. The parallel build is the same binary with Rayon splitting the assignment step across worker threads. scikit-learn routes the dominant term through a BLAS GEMM. Same arithmetic, four memory stories.

## What k-means actually is

The algorithm predates most of computing. Stuart Lloyd worked it out at Bell Labs in 1957 as a quantization scheme for pulse-code modulation (the memo wasn't formally published until 1982), and MacQueen coined the name "k-means" in 1967. A biologist sequencing ten thousand single cells, or an astronomer plotting a million stars by color and brightness, reaches for exactly this to find the groups in unlabeled data. What everyone runs today is still Lloyd's loop:

1. **Assign** each point to its closest centroid.
2. **Update** each centroid to the mean of its assigned points.

Both steps minimize the same objective, the within-cluster sum of squares:

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2$$

The assignment step minimizes $J$ over the labels $c_i$ with the centroids held fixed, and the update step minimizes it over the centroids $\mu_j$ with the labels held fixed. Neither step can increase $J$, so the loop converges to a local optimum. If you've trained anything with EM or alternating least squares, this is the same coordinate-descent shape. (The companion site's [algorithms page](https://www.nilesh42.science/bench-kmeans-rust/algorithms/) works through it as alternating minimization in full.)

<figure>
  <svg viewBox="0 0 600 205" role="img" aria-labelledby="lloyd-t lloyd-d" style="width:100%;height:auto;max-width:600px;color:inherit" xmlns="http://www.w3.org/2000/svg">
    <title id="lloyd-t">Lloyd's loop: assign, then update, then repeat</title>
    <desc id="lloyd-d">Two alternating steps. Assign sends each point to its nearest centroid with the centroids held fixed. Update moves each centroid to the mean of its points with the labels held fixed. The two steps trade back and forth, each one lowering the objective J, until the labels stop moving.</desc>
    <defs>
      <marker id="lloyd-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
      </marker>
    </defs>
    <g stroke="currentColor" stroke-width="1.5" fill="none">
      <rect x="96" y="64" width="160" height="74" rx="9"></rect>
      <rect x="344" y="64" width="160" height="74" rx="9"></rect>
      <path d="M256,86 Q300,57 344,86" marker-end="url(#lloyd-ah)"></path>
      <path d="M344,116 Q300,145 256,116" marker-end="url(#lloyd-ah)"></path>
      <path d="M16,101 L96,101" marker-end="url(#lloyd-ah)"></path>
      <path d="M504,101 L584,101" marker-end="url(#lloyd-ah)"></path>
    </g>
    <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" text-anchor="middle" aria-hidden="true">
      <text x="176" y="95" font-size="17" font-weight="600">Assign</text>
      <text x="176" y="117" font-size="11.5" opacity="0.7">point &#8594; nearest centroid</text>
      <text x="424" y="95" font-size="17" font-weight="600">Update</text>
      <text x="424" y="117" font-size="11.5" opacity="0.7">centroid &#8594; mean of its points</text>
      <text x="300" y="49" font-size="11" opacity="0.7">centroids held fixed</text>
      <text x="300" y="161" font-size="11" opacity="0.7">labels held fixed</text>
      <text x="300" y="194" font-size="12.5">each step lowers <tspan font-style="italic">J</tspan>, so the loop always converges</text>
    </g>
    <text x="12" y="92" fill="currentColor" font-family="-apple-system, system-ui, sans-serif" font-size="11" opacity="0.7" text-anchor="start" aria-hidden="true">k-means++ init</text>
    <text x="588" y="92" fill="currentColor" font-family="-apple-system, system-ui, sans-serif" font-size="11" opacity="0.7" text-anchor="end" aria-hidden="true">converged</text>
  </svg>
  <figcaption>Lloyd's loop as alternating minimization. <strong>Assign</strong> holds the centroids fixed and minimizes $J$ over the labels; <strong>Update</strong> holds the labels fixed and minimizes it over the centroids. Two convex half-steps traded back and forth, each one lowering $J$, until the labels stop moving  the same coordinate-descent shape as EM, and the reason the loop always settles into a local (not global) optimum.</figcaption>
</figure>

The cost per iteration is $O(nkd)$  every point against every centroid across every dimension, dominated by the distance computation. And that distance computation hides a matrix multiply. Stack the data into $X$ (shape $n \times d$) and the $k$ centroids into a matrix $M$ (shape $k \times d$), one centroid per row. Expanding the square,

$$\lVert x_i - \mu_j \rVert^2 = \lVert x_i \rVert^2 - 2\, x_i \cdot \mu_j + \lVert \mu_j \rVert^2,$$

the cross term $x_i \cdot \mu_j$ over all pairs at once is the $n \times k$ matrix $X M^\top$, formed by contracting over the $d$ feature dimensions  the $O(nkd)$ work lives in that contraction. The $-2$ scales that block, and the two norm terms (one cheap pass over the points, one over the centroids) wrap around it to assemble the full distance matrix. The same split is why scikit-learn is fast at scale. It hands $X M^\top$ to BLAS, the matrix-multiply kernels tuned for decades to keep data in cache and saturate SIMD units. It's also the corner my naive Rust gives back, as we'll see.

<figure>
  <img src="/images/blog/bench-kmeans-rust/convergence_random.gif" alt="Animation: a single random-seeded k-means run settles into a poor split and never recovers the true clusters." loading="lazy">
  <figcaption>A single random start dropping into a poor local optimum and staying there. The centroids find a split that locally minimizes $J$ yet never recovers the true clusters  the failure mode that turned out to explain my first run's strangest result, dissected in the accuracy section below. Reproduce it with random init in the <a href="https://www.nilesh42.science/bench-kmeans-rust/demo/">live demo</a>.</figcaption>
</figure>

## Where the speedup decays

Across the full sweep, serial Rust is a median 4.5x faster than pure-Python NumPy. That figure is a median of paired ratios  each of the 162 datasets gets one Python-time-over-Rust-time ratio, and the median is taken across those. The mean of the same ratios is about 6.6x, pulled up by the small-n cells where Rust shines, and the ratio of overall median runtimes is 4.02x. I lead with the median for the same reason you report p50 latency rather than the average: it describes the configuration you'll actually hit. Parallel Rust comes in at 5.1x paired, but hold that thought  almost all of it is the serial win with a thin thread bonus on top.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-1-runtime-vs-size.svg" alt="Two log-log runtime charts. Left, pooled over the grid: Rust is the lowest line at every size from 1k to 256k. Right, the heaviest slice (32 features, 32 clusters): scikit-learn's line dips below Rust's at n=128k and n=256k." loading="lazy">
  <figcaption>Median end-to-end runtime vs dataset size. The baked annotation on the right panel reads "scikit-learn overtakes Rust at n=128,000"  but that overtake lives only in the heaviest corner (32 features, k_max=32), one of just 4 cells out of 54 where scikit-learn wins. Pooled over the whole grid (left panel) Rust is the lowest line at every size, 256k included. The crossover is a corner case, not a general fact.</figcaption>
</figure>

Pooled across the grid, Rust runs from 0.029 s at n=1k to 4.13 s at n=256k. Its fitted slope is 0.902 (R² 0.993), close to linear and slightly under it  about what you'd expect from naive $O(nkd)$ loops picking up some cache pressure. NumPy's slope is 0.599 (R² 0.911) and scikit-learn's is a near-flat 0.249 (R² 0.831). These are descriptive fits over size-medians, not clean algorithmic exponents, which the R² values reflect. The ordering is the story. The more fixed overhead an implementation carries, the less it notices additional rows, and that is why the speedup decays.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-2-speedup-curve.svg" alt="Speedup over pure-Python NumPy vs size. Rust decays from about 14.5x at n=1k to about 3.5x at n=256k. scikit-learn starts at 0.27x and only crosses parity near n=128k." loading="lazy">
  <figcaption>Speedup over pure-Python NumPy, median of paired ratios. The two curves converge from opposite sides because they sit on opposite sides of the overhead trade. Rust's per-row arithmetic grows roughly linearly, so its lead bleeds off as n climbs  14.5x at n=1k down to about 3.5x at n=256k, never below 3x. scikit-learn carries a heavy fixed cost (library start-up, BLAS warm-up) that dwarfs the actual work at small n  0.27x there, nearly four times slower than plain NumPy  but that overhead amortizes as n grows, so it climbs to 1.80x and reaches parity near n=128k. At <code>n_init=1</code>, none of scikit-learn's small-n drag is restart cost.</figcaption>
</figure>

Whenever someone quotes a single speedup number for a rewrite, the answer depends entirely on where you sit on these two curves.

The corner scikit-learn actually wins follows directly. My first, uncontrolled run had it "overtaking Rust at n=128k," full stop; the rerun sharpens that. Pooled over the grid, Rust never loses to scikit-learn at the median, not even at 256k. scikit-learn is faster in exactly 4 of 54 grid cells, and all four have k_max=32 and n ≥ 128k. The crossover barely cares about feature count  two of those four cells are at f=2, the skinniest data in the grid. What tips the balance is rows times clusters, the $n \times k$ face of the distance computation, more than the full GEMM volume. In the heaviest slice (f=32, k=32), scikit-learn finishes in 8.0 s against serial Rust's 9.4 s at 128k, and 15.4 s against 20.8 s at 256k. At the light end Rust beats scikit-learn by 6x to 65x with no crossover in sight.

The single heaviest workload (256k rows, 32 features, k=32) is the clearest snapshot of all four at once: scikit-learn 15.36 s, parallel Rust 16.23 s, serial Rust 20.76 s, pure-Python NumPy 46.46 s. scikit-learn runs the same Lloyd iteration as everyone else; it simply hands the dominant term to a GEMM that keeps the distance block in cache and saturates the vector units. My Rust loses that corner because my loops are naive. A BLAS-backed Rust (an `ndarray` GEMM for the cross term) would very likely take it back. The ceiling was my loops, not the language.

## The parallel Rust story

The usual pitch for a Rust rewrite is "add Rayon, get 14 cores." If threads paid off, this would be the build you'd actually ship, so I swept thread counts at every size.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-4-parallel-scaling.svg" alt="Rayon speedup over serial Rust vs worker-thread count, one curve per dataset size. Gains top out near 1.3x at large n; at n=1k, 14 threads run about 28% slower than serial." loading="lazy">
  <figcaption>Rayon speedup over serial Rust by thread count, one curve per dataset size (medians of 3, k_max=8). Two things hold the gain near 1.3x. Granularity  the elbow sweep fits k = 1..K, so most fits are small-k with little arithmetic per point, too little to amortize Rayon's split/join overhead, which at n=1k turns 14 cores into a 28% penalty. And bandwidth  the data is an array of structs (<code>Vec&lt;DataPoint&gt;</code>, each point owning a heap-allocated <code>Vec&lt;f64&gt;</code>), so the inner loop pointer-chases and goes memory-bandwidth-bound. A handful of cores saturate the bus; the rest only add coordination cost. The curve plateaus rather than climbing.</figcaption>
</figure>

Peak speedup climbs with n and plateaus near 1.3x  1.318x at n=256k, 1.286x at n=64k. At the heaviest workload it's concrete: 256k/32/k=32 finishes in 16.23 s parallel against 20.76 s serial, about 1.28x. At small n the threads turn harmful. n=1k peaks at 1.09x with a single worker and falls to 0.725x at 14 threads, a 28% penalty for spinning up all the cores. This is why parallel Rust's grid-wide median wall time (0.197 s) barely edges serial (0.201 s)  across most of the grid the threads add nothing, and on the smallest cells they cost. The 5.1x-vs-Python paired figure is mostly the serial 4.5x with the modest large-n thread gain riding on top.

One myth to retire: none of this is about Python's GIL. NumPy releases the GIL inside its C loops, which is exactly why BLAS-backed calls can use multiple cores. The pure-Python baseline stays on one core simply because a serial Lloyd loop never asks for more. The GIL is the wall when you point `threading` at CPU-bound *Python* code, and that's not what's happening in any of these four ways. (The [benchmarks page](https://www.nilesh42.science/bench-kmeans-rust/benchmarks/) walks the thread-scaling curves in more depth.)

## Memory

Median sampled peak RSS per 1,000 rows: serial Rust 0.61 MB, parallel Rust 0.73 MB, NumPy 7.41 MB, scikit-learn 12.63 MB. That puts serial Rust about 11x leaner than NumPy and 22x leaner than scikit-learn. Parallel Rust carries a small premium for per-thread bookkeeping but stays in the same league. Unlike the speed gap, no size flips this ranking.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-3-memory.svg" alt="Left: peak memory per 1,000 samples - Rust 0.61 MB, NumPy 7.41, scikit-learn 12.63. Right: peak RSS vs size on log-log axes - Rust is the lowest line throughout, and NumPy overtakes scikit-learn at the largest sizes." loading="lazy">
  <figcaption>Sampled peak RSS, normalized per 1,000 samples (left) and absolute by size (right). Rust stays lowest everywhere. The surprise is the right panel  NumPy, flat at small n, eventually climbs past scikit-learn, because NumPy reallocates the full $n \times k$ distance matrix every iteration while scikit-learn's chunked kernels never materialize it whole. A heavy fixed base loses to per-iteration allocation once n is large enough.</figcaption>
</figure>

The mechanism is the flip side of vectorization. In Python, "vectorized" concretely means "allocate a big array and let C fill it"  the NumPy engine materializes an $n \times k$ float64 distance matrix every single iteration, and that array is the memory bill. That allocation is why NumPy's footprint eventually overtakes scikit-learn's at the largest sizes even though scikit-learn carries a far heavier fixed base of interpreter and imports. You can read it in the heaviest workload: at 256k×32, k=32, scikit-learn peaks at 795 MB and NumPy at 924 MB, while both Rust builds sit near 190 MB (190 parallel, 194 serial). Rust never builds the distance matrix at all  its assignment step walks each point against the centroids holding only a running nearest-index, so peak memory tracks the data itself.

A correction to my earlier write-up, where I'd guessed wrong about the cause. The Rust memory win has nothing to do with a tight cache-local matrix. The layout is an array of structs with a heap `Vec<f64>` and a string id per point. The advantage comes entirely from never allocating the distance matrix. A flat contiguous `Vec<f64>` is the optimization I left on the table, and it would help the parallel story and the large-n slope too.

## Accuracy and initialization

The rerun changes a conclusion here. In my first pass the hand-rolled implementations used random init while scikit-learn used its default ten restarts, and Rust posted the worst clusterings of the four (ARI 0.66 against scikit-learn's 1.0). I briefly believed the rewrite traded accuracy for speed. The gap was the experiment design.

With every implementation on the same k-means++ init, the median ARI is 1.00 for all four. Serial and parallel Rust are bit-identical in quality  the same seed reaches the same answer whether one thread or fourteen does the arithmetic. The means barely separate (scikit-learn 0.9988, Python 0.9802, Rust 0.9742), and internal metrics agree (silhouette 0.93/0.92/0.92; Davies-Bouldin 0.10/0.18/0.19).

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-5-quality-pareto.svg" alt="Two panels. Left, mean ARI vs runtime: all implementations sit between 0.974 and 0.999, essentially tied. Right, worst-case (minimum) ARI: scikit-learn at 0.958 sits above Rust and Python at 0.834." loading="lazy">
  <figcaption>Clustering quality vs runtime. On mean ARI (left) all four tie. Worst-case ARI (right) is the one axis where scikit-learn keeps daylight  a floor of 0.958 against 0.834 for Rust and Python.</figcaption>
</figure>

scikit-learn's remaining edge is that worst case, a gap of 0.12 on the hardest configs (its 0.958 floor against 0.834 in the figure). The natural suspect is its greedy k-means++ variant, which tries several candidate seeds at each step and keeps the best, buying extra robustness. (Convergence tolerance and empty-cluster handling differ too, so the gap isn't all seeding.) It buys a sturdier start, not a faster engine.

The k-means++ idea, due to Arthur and Vassilvitskii in 2007, is worth knowing on its own. Rather than dropping k seeds uniformly at random, let $D(x)$ be the distance from $x$ to the nearest seed already chosen; then pick each new seed with probability $p(x)$ proportional to $D(x)^2$:

$$p(x) \propto D(x)^2$$

Far-away regions get seeded, clumps don't, and a single start usually lands near a good basin (the paper proves an $O(\log k)$ bound on expected cost). In a standalone pure-Python ablation with 10 seeds per config, switching random init to k-means++ cut final inertia by 37% to 54%, and the benefit grows with k and dimension.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-6-init-study.svg" alt="Bar chart: k-means++ reduces final inertia versus random init by 37.2%, 53.1%, and 54.4% across three configurations of increasing size, dimension, and cluster count." loading="lazy">
  <figcaption>Inertia reduction from switching random init to k-means++ (37.2%, 53.1%, 54.4%; 10 seeds per config). The cheapest improvement in this whole project, and it requires no rewrite at all  the gain widens as size, dimension, and cluster count grow.</figcaption>
</figure>

<figure>
  <img src="/images/blog/bench-kmeans-rust/convergence_kpp.gif" alt="Animation: k-means++ spreads its seeds far apart via squared-distance sampling, drops into the right basins, and converges in a handful of iterations." loading="lazy">
  <figcaption>k-means++ seeding in motion. Each new center is sampled far from the existing ones via the $D(x)^2$ rule, so the run starts near the right basins and converges in a few iterations  contrast the wandering random start at the top of the post. Toggle init in the <a href="https://www.nilesh42.science/bench-kmeans-rust/demo/">live demo</a> to watch the spread happen.</figcaption>
</figure>

## What no rewrite fixes

k-means carves space into convex Voronoi cells around its centroids, so it assumes roughly round, similarly sized clusters. Hand it two concentric rings and all four ways, in every language, fail identically.

<figure>
  <img src="/images/blog/bench-kmeans-rust/convergence_circles.gif" alt="Animation: k-means slices two concentric rings into pie wedges instead of separating inner ring from outer." loading="lazy">
  <figcaption>Two concentric rings, pie-sliced into wedges. Nearest-centroid assignment can't wrap a cell around another cluster, so the inner ring never separates from the outer  the model's assumption showing through, not an implementation bug. Both the WASM and JS runs in the <a href="https://www.nilesh42.science/bench-kmeans-rust/demo/">live demo</a> fail this the same way.</figcaption>
</figure>

## Summary

If your bottleneck is a NumPy k-means at small to medium `n`, the reflex is vindicated: hand-rolled serial Rust runs about 4.5x faster at the median, uses an order of magnitude less memory, and that memory edge doesn't erode as the data grows. Push to large n *and* large k and tuned linear algebra takes over  rewrite to beat your interpreter, not BLAS, and if you need that corner too, reach for a flat-matrix, GEMM-backed Rust before reaching for more threads. Parallel Rust was the smallest lever in the whole project (1.32x at best, a penalty at small n). The biggest lever was free: matching the initialization closed the accuracy gap entirely and cut inertia by up to half.

<div style="overflow-x:auto" markdown="1">

| Situation | Reach for |
|---|---|
| Small/medium NumPy k-means bottleneck | Serial Rust: ~4.5x faster, ~11x leaner |
| Very large n × k | scikit-learn, or a BLAS-backed flat-matrix Rust |
| Accuracy matters most | k-means++ on any of the four; it closed the gap for free |
| Memory is the constraint | Rust (serial or parallel), at every size |
| Hoping threads will rescue it | Measure granularity first; Rayon bought 1.32x here and hurt at small n |

</div>

All four implementations, the sweep harness, and the full Plotly dashboards are in the repo. The [companion project site](https://www.nilesh42.science/bench-kmeans-rust/) ("K-Means, four ways") walks the same comparison across [algorithms](https://www.nilesh42.science/bench-kmeans-rust/algorithms/), [benchmarks](https://www.nilesh42.science/bench-kmeans-rust/benchmarks/), and [live demo](https://www.nilesh42.science/bench-kmeans-rust/demo/) pages, and the [project page](/portfolio/bench-kmeans-rust/) has the shorter summary. Clone it and watch where the crossover lands on your own hardware  the exponents above belong to my CPU's cache hierarchy and BLAS build as much as to the algorithm.


<style>
a.btn-soft {
  display: inline-block;
  margin: 0 0.5rem 0.55rem 0;
  padding: 0.5em 1.05em;
  font-size: 0.92rem;
  font-weight: 500;
  line-height: 1.3;
  color: inherit;
  text-decoration: none;
  border-radius: 8px;
  border: 1px solid rgba(128, 128, 128, 0.30);
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
  background: rgba(128, 128, 128, 0.06);
  background: color-mix(in srgb, currentColor 5%, transparent);
  transition: background-color .18s ease, border-color .18s ease, color .18s ease;
}
a.btn-soft:hover {
  text-decoration: none;
  border-color: rgba(128, 128, 128, 0.55);
  border-color: color-mix(in srgb, currentColor 42%, transparent);
  background: rgba(128, 128, 128, 0.12);
  background: color-mix(in srgb, currentColor 11%, transparent);
}
a.btn-soft--primary {
  font-weight: 600;
  border-color: rgba(128, 128, 128, 0.55);
  border-color: color-mix(in srgb, currentColor 42%, transparent);
}
</style>

[Race WASM vs JS in the live demo](https://www.nilesh42.science/bench-kmeans-rust/demo/){: .btn-soft .btn-soft--primary} [Read the project write-up](https://www.nilesh42.science/bench-kmeans-rust/){: .btn-soft} [Browse the code](https://github.com/nilesh-patil/bench-kmeans-rust){: .btn-soft}