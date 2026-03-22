---
layout: single
title: "parallel Rust k-means - benchmarking a clustering rewrite against NumPy and scikit-learn"
date: 2026-03-22T20:00:00+05:30
last_modified_at: 2026-06-15T02:36:16+05:30
categories: [blog]
tags: [rust, python, machine-learning, clustering, performance, parallelism]
excerpt: "Four k-means engines on one grid: hand-rolled NumPy, serial Rust, a Rayon-parallel Rust, and scikit-learn. The hand-rolled Rust wins 15x at a thousand points, 3.5x at a quarter-million, and loses one corner outright. Where the speedup went, and what parallelism and a matched init buy."
measurement: "15x at a thousand points"
math: true
redirect_from:
  - /posts/just-rewrite-it-in-rust-kmeans/
header:
  overlay_image: /images/blog/headers/parallel-rust-kmeans-rewrite.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/parallel-rust-kmeans-rewrite.jpg
---

*Update, June 2026: every number and figure below comes from a 648-run rerun measured on 9 June 2026 ([results CSV](https://github.com/nilesh-patil/bench-kmeans-rust/blob/master/results/benchmark_results_20260609_112255.csv)), after this post's date. The earlier run it supersedes did not match initialization across implementations; that [earlier write-up](https://github.com/nilesh-patil/bench-kmeans-rust/blob/a2e82a5d37717b662a1b3b96e16435a85f436b1c/docs/index.md) is still in the repo history.*

Open the [live demo](https://www.nilesh42.science/bench-kmeans-rust/demo/), pick a point distribution, and hit run. The first pass clusters your points with the hand-rolled Rust k-means from this post, compiled to WebAssembly; the second is a pure-JavaScript k-means on the *same* data and the *same* seed, timed beside it with `performance.now()` on your machine. I ran the native version of that race against NumPy and scikit-learn. At n=1,000, a full elbow sweep (load the CSV, fit every k from 1 to K, write results) takes my hand-rolled NumPy k-means 0.43 s and scikit-learn 1.57 s; the Rust binary does it in 0.029 s, roughly 15x over NumPy and 54x over BLAS-backed scikit-learn. Those are ratios of the median runtimes at n=1k; the median of paired per-dataset ratios there is 14.5x. All four implementations start from the same k-means++ initialization. By a quarter-million points the 15x has shrunk to 3.5x, and in one corner of the grid scikit-learn wins.

The four run one algorithm, Lloyd's assign-then-update loop:

1. **Hand-rolled NumPy** - Lloyd's loop written in Python, with the array arithmetic done by NumPy. The figures label it "Python (NumPy)" or "pure-Python" and the results CSV calls it Python; all of those mean this implementation.
2. **Serial Rust** - explicit loops, no interpreter, no per-operation allocations.
3. **Parallel Rust** - the *same* binary run with `--parallel --threads 0`, which turns on Rayon. It ran the whole grid like the other three.
4. **scikit-learn** at `n_init=1` - one start, like the rest.

<div class="notice--info" markdown="1">
**Four ways at a glance** (648 runs, Apple M4 Max, all implementations on identical k-means++ init):

<div style="overflow-x:auto" markdown="1">

| Implementation | Median wall time | Speedup vs NumPy | Median RSS / 1k rows | Median ARI |
|---|---|---|---|---|
| **Hand-rolled NumPy** | 0.806 s | 1.0x (baseline) | 7.41 MB | 1.00 |
| **Serial Rust** | 0.201 s | **4.5x** | **0.61 MB** | 1.00 |
| **Parallel Rust (Rayon)** | 0.197 s | **5.1x** | 0.73 MB | 1.00 |
| **scikit-learn** (`n_init=1`) | 1.843 s | 0.44x | 12.63 MB | 1.00 |

</div>

Serial Rust's lead shrinks with size, from 15x to 3.5x. Rayon adds at most 1.32x over serial, and Rust uses the least memory at each size.

*How the speedup column is computed.* Each figure is a median of paired per-dataset ratios (NumPy time over the other implementation's time, one ratio per dataset), taken from [`specs/analysis_facts.md`](https://github.com/nilesh-patil/bench-kmeans-rust/blob/master/specs/analysis_facts.md) in the repo. scikit-learn's 0.44x also equals the ratio of the overall medians (0.806 / 1.843), but it averages over a reversal: 0.27x at n=1k, 1.80x at n=256k.
</div>

## Lloyd's loop and the distance block

Stuart Lloyd worked the algorithm out at Bell Labs in 1957 as a quantization scheme for pulse-code modulation, though the memo wasn't formally published until 1982, and MacQueen coined the name "k-means" in 1967. NumPy, Rust and scikit-learn all still run his loop:

1. **Assign** each point to its closest centroid.
2. **Update** each centroid to the mean of its assigned points.

Both steps minimize the same objective, the within-cluster sum of squares:

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2$$

The assignment step minimizes $J$ over the labels $c_i$ with the centroids held fixed, and the update step minimizes it over the centroids $\mu_j$ with the labels held fixed. Neither step can increase $J$, so the loop converges to a local optimum. (The companion site's [algorithms page](https://www.nilesh42.science/bench-kmeans-rust/algorithms/) works through it as alternating minimization in full.)

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
  <figcaption>Lloyd's loop. <strong>Assign</strong> minimizes $J$ over the labels with the centroids fixed; <strong>Update</strong> minimizes it over the centroids with the labels fixed. The loop stops when the labels stop moving.</figcaption>
</figure>

The cost per iteration is $O(nkd)$: each point against each centroid across each dimension. Almost all of it is the distance computation, and the distance computation contains a matrix multiply. Stack the data into $X$ (shape $n \times d$) and the $k$ centroids into a matrix $M$ (shape $k \times d$), one centroid per row. Expanding the square,

$$\lVert x_i - \mu_j \rVert^2 = \lVert x_i \rVert^2 - 2\, x_i \cdot \mu_j + \lVert \mu_j \rVert^2,$$

the cross term $x_i \cdot \mu_j$ over all pairs at once is the $n \times k$ matrix $X M^\top$, formed by contracting over the $d$ feature dimensions. The $O(nkd)$ work is in that contraction; the two norm terms are one cheap pass over the points and one over the centroids.

The implementations differ in how they produce those $n \times k$ distances. The NumPy version allocates the full matrix and fills it one centroid column at a time with `np.linalg.norm(X - centroids[k], axis=1)`. Serial Rust runs explicit loops and keeps only a running nearest index per point, so the matrix never exists. Parallel Rust is the same code with Rayon splitting the assign and update steps across worker threads. scikit-learn routes the cross term through a BLAS GEMM.

## The grid

One end-to-end CLI run per configuration: load CSV, fit k = 1..K, write CSV, exit. I time the whole subprocess, process launch included, because that is what a pipeline pays per call. The grid is 9 sizes (1k to 256k, doubling) × 3 feature counts (2/8/32) × 2 cluster budgets (k_max 8 or 32) × 3 repeats, on synthetic Gaussian blobs: 162 dataset cells × 4 implementations = 648 runs, on an Apple M4 Max. None of the 648 exited with an error and none is missing ARI or NMI.

The control I care most about is initialization. Every implementation seeds with k-means++, and scikit-learn runs at `n_init=1`. This is a rerun of an [earlier experiment](https://github.com/nilesh-patil/bench-kmeans-rust/blob/a2e82a5d37717b662a1b3b96e16435a85f436b1c/docs/index.md) that didn't control init, and matching it changes one headline result, in the quality section.

Metrics are subprocess wall time, sampled peak RSS (polled every 10 ms, so it can miss brief spikes that a kernel max-RSS would catch), and ARI/NMI against the ground-truth blob labels. For reference, the versions the repo pins in `pixi.lock` are rustc 1.87.0, NumPy 2.4.5, scikit-learn 1.8.0 and OpenBLAS 0.3.33; the results CSV does not record versions per run.

## Wall time against NumPy

Serial Rust is the NumPy version's arithmetic written as explicit loops: no interpreter dispatch, no per-operation allocations, no distance matrix. Across the full sweep that is worth a median 4.5x.

The 4.5x is a median of paired ratios, one NumPy-time-over-Rust-time ratio for each of the 162 datasets. The mean of the same ratios is about 6.6x, pulled up by the small-n cells where Rust does best, and the ratio of overall median runtimes is 4.02x. I lead with the paired median because those small cells dominate the mean.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-1-runtime-vs-size.svg" alt="Two runtime charts comparing implementations against dataset size, pooled over the grid on the left and the heaviest slice on the right." loading="lazy">
  <figcaption>Median end-to-end runtime vs dataset size, pooled over the whole grid (left) and for the heaviest slice alone, 32 features and k_max=32 (right).</figcaption>
</figure>

Pooled across the grid, Rust is the lowest line at each size and also the steepest: for 256x the rows its time grows about 144x (0.029 s to 4.13 s), where NumPy's grows 29x and scikit-learn's 4.4x. A log-log fit over the size-medians gives slopes of 0.902, 0.599 and 0.249, but NumPy and scikit-learn follow a power law only loosely (R² 0.911 and 0.831), so I don't read any of the three as an algorithmic exponent. What they do rank is fixed overhead: the flatter the slope, the larger the share of a run that is start-up and per-call cost, and the less an extra row matters. Rust has the least overhead to amortize, so its lead shrinks as n climbs.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-2-speedup-curve.svg" alt="Line chart of speedup over NumPy against dataset size: a decaying Rust curve and a rising scikit curve cross parity at large n." loading="lazy">
  <figcaption>Speedup over the hand-rolled NumPy (the axis label says pure-Python) against dataset size, as a median of paired ratios, with parity marked.</figcaption>
</figure>

Serial Rust goes from 14.5x at n=1k to about 3.5x at n=256k and stays above 3x throughout. scikit-learn comes from the other side. At n=1k it sits at 0.27x, nearly four times slower than the hand-rolled NumPy, because its fixed cost (library start-up, BLAS warm-up) dwarfs the fitting work. That cost amortizes as n grows, and scikit-learn reaches parity near n=128k and 1.80x at 256k. None of the small-n drag is restart cost, since it runs at `n_init=1`.

## Where scikit-learn wins: four cells of 54

scikit-learn has the flattest slope of the four because most of its cost is fixed and its per-row work is the cheapest, and at the top of the grid that catches my loops. Pooled over the grid, Rust still beats scikit-learn at the median at each size, 256k included. Cell by cell, scikit-learn is faster in 4 of the 54 (size, features, k_max) cells, and all four have k_max=32 and n ≥ 128k. Feature count barely matters: two of the four are at f=2, the narrowest data in the grid. What tips the balance is rows times clusters, the $n \times k$ face of the distance computation, more than the full GEMM volume. In the heaviest slice (f=32, k_max=32), scikit-learn finishes in 8.0 s against serial Rust's 9.4 s at 128k. At the light end Rust beats scikit-learn by 6x to 65x.

The single heaviest workload (256k rows, 32 features, k_max=32) puts the four in this order: scikit-learn 15.36 s, parallel Rust 16.23 s, serial Rust 20.76 s, hand-rolled NumPy 46.46 s. scikit-learn spent 24.59 s of CPU time inside its 15.36 s of wall time, 1.65 effective cores where serial Rust used 0.98, so part of its win there comes from using more than one core. The rest I put down to the GEMM keeping the distance block in cache and filling the vector units, which my scalar loops don't do. I haven't built the comparison that would test this, a Rust version with an `ndarray` GEMM for the cross term.

## Fourteen threads

Parallel Rust is one flag: `--parallel --threads 0` lets Rayon use all 14 cores. The usual pitch for a Rust rewrite is "add Rayon, get 14 cores," so I swept thread counts (1, 2, 4, 8, 14) at each of the nine sizes.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-4-parallel-scaling.svg" alt="Rayon speedup over serial Rust against thread count, one curve per dataset size, plateauing at large n and below parity at small n." loading="lazy">
  <figcaption>Rayon speedup over serial Rust by thread count, one curve per dataset size (medians of 3, k_max=8). Values below 1 mean the parallel run was slower than serial.</figcaption>
</figure>

On 14 cores the loop runs barely faster than on one. The best speedup at any size is 1.318x, at n=256k with 8 threads, and 14 threads there is slower than 8. On the main grid the heaviest workload (256k/32/k_max=32) gets about 1.28x out of 1.31 effective cores. At small n the threads cost time: n=1k falls to 0.725x at 14 threads.

Across the grid the parallel build's median wall time is 0.197 s against 0.201 s serial, with a median of 1.87 effective cores. The 5.1x-over-NumPy paired figure is mostly the serial 4.5x plus the modest large-n thread gain.

Two things would explain a ceiling near 1.3x. The elbow sweep fits k = 1..K, so most fits are small-k with little arithmetic per point, too little to pay back Rayon's split/join overhead. And the data is an array of structs (`Vec<DataPoint>`, each point owning a heap-allocated `Vec<f64>`), so the inner loop chases a pointer per point and is probably bound by memory bandwidth: a few cores saturate it and the others add coordination cost.

Python's GIL plays no part in any of this. NumPy releases the GIL inside its C loops, which is why BLAS-backed calls can use several cores. The hand-rolled NumPy baseline sits at a median 0.91 effective cores because a serial Lloyd loop has no work to hand to a second one. (The [benchmarks page](https://www.nilesh42.science/bench-kmeans-rust/benchmarks/) walks the thread-scaling curves in more depth.)

## Peak memory

Per 1,000 rows, the median sampled peak RSS is 0.61 MB for serial Rust against 7.41 MB for NumPy and 12.63 MB for scikit-learn, about 12x and 21x leaner, and the parallel build pays a small premium at 0.73 MB. Rust is lowest at each size in the grid.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-3-memory.svg" alt="Two memory charts: peak memory per 1,000 samples on the left, and peak RSS against dataset size on the right." loading="lazy">
  <figcaption>Sampled peak RSS, normalized per 1,000 samples (left) and absolute by size (right). In the right panel the NumPy line crosses above scikit-learn at large n.</figcaption>
</figure>

The other two do swap places. NumPy starts below scikit-learn and climbs past it at large n, because it reallocates the full $n \times k$ distance matrix on each iteration while scikit-learn's chunked kernels work through it in pieces. In the heaviest workload (256k×32, k_max=32) scikit-learn peaks at 795 MB and NumPy at 924 MB, and both Rust builds sit near 190 MB (190 parallel, 194 serial). Rust's assignment step walks each point against the centroids holding only a running nearest index, so its peak memory tracks the data.

The [earlier write-up](https://github.com/nilesh-patil/bench-kmeans-rust/blob/a2e82a5d37717b662a1b3b96e16435a85f436b1c/docs/index.md) credited this to a flat `Vec<f64>`, and I'd guessed wrong. The layout is an array of structs with a heap `Vec<f64>` and a string id per point, which is poor for cache locality, so the memory advantage comes entirely from never allocating the distance matrix. A flat contiguous `Vec<f64>` is an optimization I left undone, and I'd expect it to help the thread scaling and the large-n slope too.

## Clustering quality with matched init

This is the result the rerun changed. In the earlier run the hand-rolled implementations used random init while scikit-learn used its default ten restarts, and Rust posted the worst clusterings of the four (ARI 0.66 against scikit-learn's 1.0). Taken at face value that is a rewrite trading accuracy for speed, when it was an artifact of how I had set up the comparison.

<figure>
  <img src="/images/blog/bench-kmeans-rust/convergence_random.gif" alt="Animation: a single random-seeded k-means run settles into a poor split and never recovers the true clusters." loading="lazy">
  <figcaption>A single random start settling into a poor local optimum. The split locally minimizes $J$ and the true clusters are never recovered. Reproduce it with random init in the <a href="https://www.nilesh42.science/bench-kmeans-rust/demo/">live demo</a>.</figcaption>
</figure>

With all four on the same k-means++ init, the median ARI is 1.00 for each, and serial and parallel Rust match in each quality column of the run summary. The means barely separate (scikit-learn 0.9988, NumPy 0.9802, Rust 0.9742). The count of runs that reach ARI ≥ 0.999 separates them more: 156 of 162 for scikit-learn, 113 for NumPy, 106 for Rust.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-5-quality-pareto.svg" alt="Two panels: mean ARI against runtime on the left with implementations nearly tied; minimum ARI on the right with scikit above Rust and Python." loading="lazy">
  <figcaption>Clustering quality against runtime: mean ARI (left) and minimum ARI (right). The minimum is 0.958 for scikit-learn and 0.834 for Rust and NumPy.</figcaption>
</figure>

scikit-learn's remaining edge is that worst case, and at least part of it comes from its greedy k-means++ variant, which tries several candidate seeds at each step and keeps the best. Convergence tolerance and empty-cluster handling differ too, so the gap isn't all seeding.

k-means++ itself, due to Arthur and Vassilvitskii in 2007, changes the seeding and leaves the loop alone. Let $D(x)$ be the distance from $x$ to the nearest seed already chosen; each new seed is then picked with probability $p(x)$ proportional to $D(x)^2$:

$$p(x) \propto D(x)^2$$

Far-away regions get seeded, clumps mostly don't, and a single start usually lands near a good basin (the paper proves an $O(\log k)$ bound on expected cost). In a standalone ablation on the Python implementation with 10 seeds per config, switching random init to k-means++ cut final inertia by 37% to 54%, and the cut grows with k and dimension.

<figure>
  <img src="/images/blog/bench-kmeans-rust/figure-6-init-study.svg" alt="Bar chart: k-means++ cuts final inertia versus random init across three growing configurations of increasing size, dimension, and cluster count." loading="lazy">
  <figcaption>Reduction in final inertia from switching random init to k-means++, for three configurations of growing size, dimension, and cluster count: 37.2%, 53.1%, 54.4% (10 seeds each).</figcaption>
</figure>

<figure>
  <img src="/images/blog/bench-kmeans-rust/convergence_kpp.gif" alt="Animation: k-means++ spreads its seeds far apart via squared-distance sampling, drops into the right basins, and converges in a handful of iterations." loading="lazy">
  <figcaption>k-means++ seeding in motion. Each new center is sampled far from the existing ones by the $D(x)^2$ rule, and the run converges in a few iterations. Toggle init in the <a href="https://www.nilesh42.science/bench-kmeans-rust/demo/">live demo</a> to compare it with the random start above.</figcaption>
</figure>

## Concentric rings

k-means carves space into convex Voronoi cells around its centroids, so it assumes roughly round, similarly sized clusters. Give it two concentric rings and it cuts them into wedges, however fast the implementation.

<figure>
  <img src="/images/blog/bench-kmeans-rust/convergence_circles.gif" alt="Animation: k-means slices two concentric rings into pie wedges instead of separating inner ring from outer." loading="lazy">
  <figcaption>Two concentric rings, cut into wedges. Nearest-centroid assignment can't wrap one cell around another cluster, so the inner ring is never separated from the outer. The WASM and JS runs in the <a href="https://www.nilesh42.science/bench-kmeans-rust/demo/">live demo</a> both do this.</figcaption>
</figure>

## Choosing between the four

If the bottleneck is a NumPy k-means at small to medium `n`, hand-rolled serial Rust runs about 4.5x faster at the median on roughly a twelfth of the memory, and the memory advantage holds as the data grows. At large n *and* large k, tuned linear algebra wins; if I needed that corner I would try a flat-matrix, GEMM-backed Rust before adding threads. Rayon was the smallest gain in the project (1.32x at best, a penalty at small n). Matching the initialization was the largest: it closed the accuracy gap, and in the ablation k-means++ cut inertia by up to half.

<div style="overflow-x:auto" markdown="1">

| Situation | Reach for |
|---|---|
| Small/medium NumPy k-means bottleneck | Serial Rust: ~4.5x faster, ~12x leaner |
| Very large n × k | scikit-learn, or a BLAS-backed flat-matrix Rust |
| Accuracy matters most | k-means++ on any of the four; it closed the gap |
| Memory is the constraint | Rust (serial or parallel), at each size in the grid |
| Hoping threads will rescue it | Measure granularity first; Rayon bought 1.32x here and hurt at small n |

</div>

The result I would not bet on transferring is the crossover corner. The slopes above reflect my M4 Max's cache hierarchy and BLAS build as much as the algorithm, so the n where scikit-learn overtakes naive Rust will land somewhere else on your machine. The four implementations, the sweep harness, and the full Plotly dashboards are in the repo.

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
