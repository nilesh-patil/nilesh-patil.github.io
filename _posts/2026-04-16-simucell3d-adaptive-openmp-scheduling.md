---
layout: single
title: "Adaptive OpenMP scheduling in SimuCell3D for tissue mechanics simulations"
date: 2026-04-16T10:00:00-04:00
last_modified_at: 2026-04-16T10:00:00-04:00
categories: [blog]
tags: [cpp, hpc, openmp, simucell3d, computational-biology, profiling]
excerpt: "A tissue simulator's threads sat pegged at 800% while a third of that time was idle at a barrier. Measuring the imbalance and teaching the scheduler to adapt bought 2x the cells per compute budget."
math: true
header:
  overlay_image: /images/blog/headers/simucell3d.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/simucell3d.jpg
---

<style>
/* Scoped to this post: light hero needs a dark title instead of the theme's white */
.page__hero--overlay .page__title,
.page__hero--overlay .page__lead,
.page__hero--overlay .page__meta,
.page__hero--overlay .page__meta a { color: #13233a !important; text-shadow: none !important; }
</style>

## A sheet of cells decides to fold

A flat layer of identical cells can stay one layer thick. It can also buckle, pile into a stratified mass, or roll itself into a closed tube. The cells' chemistry alone does not decide. The shape falls out of mechanics: each cell pushes back on its volume, pulls its surface taut under cortical tension, and sticks to its neighbors with some adhesion, and the tissue settles into whatever geometry balances those forces. So if you want to understand how a gut tubule closes or how a spheroid hollows into a fluid-filled vesicle, you are not tracking a gene. You are solving for the shape a few thousand deformable surfaces fall into when they are squeezed against each other in three dimensions. That is hard to measure in a real embryo and harder to reason about by hand, which is why people simulate it.

<figure>
<svg viewBox="0 0 365 220" role="img" aria-labelledby="cell-t cell-d" style="width:100%;height:auto;max-width:365px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="cell-t">A cell as a closed triangulated surface</title>
<desc id="cell-d">A single cell drawn as a closed triangulated mesh, with a faint adjacent cell pressed against one side to indicate cell-cell contact.</desc>
<g stroke="currentColor" fill="none" opacity="0.35" stroke-width="1.4">
<path d="M284,54 Q328,48 346,94 Q358,130 330,168 Q298,190 266,178 Q234,160 230,116 Q233,76 284,54 z" fill="currentColor" fill-opacity="0.05"></path>
<path d="M284,54 L330,168 M346,94 L266,178" opacity="0.7"></path>
</g>
<g stroke="currentColor" stroke-width="1.6" fill="currentColor" fill-opacity="0.07">
<path d="M120,30 Q60,40 38,96 Q22,150 70,186 Q120,212 178,196 Q236,180 246,122 Q252,70 206,42 Q166,18 120,30 z"></path>
</g>
<g stroke="currentColor" stroke-width="1.1" fill="none" opacity="0.85">
<path d="M120,30 L70,90 M70,90 L38,96 M70,90 L60,150 M60,150 L70,186 M70,90 L130,108 M130,108 L60,150 M60,150 L120,170 M120,170 L70,186 M120,170 L178,196 M130,108 L120,170 M130,108 L190,96 M190,96 L246,122 M190,96 L206,42 M206,42 L120,30 M120,30 L130,108 M190,96 L178,160 M178,160 L246,122 M178,160 L120,170 M178,160 L178,196 M190,96 L130,108"></path>
</g>
<g fill="currentColor" stroke="none">
<circle cx="120" cy="30" r="3"></circle>
<circle cx="38" cy="96" r="3"></circle>
<circle cx="70" cy="90" r="3"></circle>
<circle cx="60" cy="150" r="3"></circle>
<circle cx="70" cy="186" r="3"></circle>
<circle cx="130" cy="108" r="3"></circle>
<circle cx="120" cy="170" r="3"></circle>
<circle cx="178" cy="196" r="3"></circle>
<circle cx="178" cy="160" r="3"></circle>
<circle cx="190" cy="96" r="3"></circle>
<circle cx="246" cy="122" r="3"></circle>
<circle cx="206" cy="42" r="3"></circle>
</g>
<g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" aria-hidden="true">
<text x="120" y="214" font-size="12" font-weight="600" text-anchor="middle">cell: closed triangulated surface</text>
<text x="298" y="214" font-size="11" opacity="0.7" text-anchor="middle">neighbor</text>
<text x="270" y="107" font-size="11" opacity="0.7" text-anchor="middle">contact</text>
</g>
</svg>
<figcaption>Each cell in SimuCell3D is a closed triangulated surface, on average 121 nodes and 238 faces; neighbors interact through a contact model handling adhesion and volumetric exclusion. Cell shape is not constrained by the representation.</figcaption>
</figure>

[SimuCell3D](https://git.bsse.ethz.ch/iber/Publications/2024_runser_simucell3d) takes the literal version of that picture. It is a deformable cell model: every cell is a closed triangulated surface, around 121 nodes and 238 faces on average, free to deform into whatever shape the forces demand. Each membrane carries an energy potential with four terms - internal pressure, area elasticity, surface tension, and bending stiffness - and the cells interact through a contact model that handles adhesion and volumetric exclusion. On top of that the engine models growth, division on a volume threshold, an extracellular matrix, fluid-filled lumens, nuclei, plus the apical/basal/lateral mechanics of polarized epithelia. Steve Runser, Roman Vetter, and Dagmar Iber built it at ETH Zurich's D-BSSE and published it in Nature Computational Science in 2024. When I first ran it, what struck me was the arithmetic underneath all those features: in the paper's benchmark, on an eight-core Intel Xeon (W-2245, 3.9 GHz), it grows a tissue from a single cell to 125,000 cells in a day of compute. The program is built to push into the tens of thousands of cells, and at that resolution the number that tends to bind is how many cells you can reach before the clock runs out.

On eight cores, under the same wall-clock budget, my fork of SimuCell3D grew a simulated tissue to **19,958 cells** where the v1.0 baseline stopped at **9,693**. Twice the tissue for the same compute, and the lever was not a faster integrator or a smaller time step. It was the order in which threads pick up work. `top` had been lying to me the whole time: eight cores pegged near 800% utilization, the textbook picture of a CPU-bound job. A tracing profiler told the real story. On the worst steps, close to a third of that "busy" time was threads standing at a barrier, waiting for one overloaded thread to finish an oversized slice.

That idle third is invisible to `top` because it reports *aggregate* CPU time. 800% across eight cores looks identical whether one thread sprints while seven idle or all eight share the load evenly. Only a per-thread trace tells you which. If you train models, you already know this failure by another name: a data-parallel step where one worker drew all the long sequences while the rest sat idle at the all-reduce barrier.

**What this is, and what it is not.** This is a writeup of my [fork](https://github.com/nilesh-patil/simucell3d) of SimuCell3D. Past a few bug fixes to the original, it covers three things: where the load imbalance came from, how I measured it with one cheap number, and what it took to make the scheduler adapt as the tissue grows. It is *not* a clean scheduler-only ablation. The same fork also rewrote contact detection, and I have not separated the two effects, a caveat I return to in full near the end. It is also a single-node story: I label the two runs **RUN01** (eight cores, the headline) and **RUN02** (seven cores, a second reproduction on different hardware), with no distributed or MPI scaling. The post runs long, so treat the section headers as signposts and skip to "The numbers" if that is what you came for.

> **TL;DR.** Static OpenMP scheduling handed every thread an equal slice of cells regardless of per-cell cost, so whichever thread drew the heavy cells kept them while the rest idled at the barrier. I built a cheap workload estimator, wired it to per-loop schedule selection, and let it adapt as the tissue grows. On the same eight cores and stopping point, adaptive in RUN01 reached **19,958 cells** to v1's **9,693**. At matched cell counts, median adaptive throughput runs **~2.0-2.6x higher** across most of the range, peaking near **2.75x**. RUN02, on different hardware and carried further, reproduced the result and extended it to **26,534 cells** against v1's **12,851**, the same 2.06x ratio as RUN01. The one caveat: this run bundles the scheduling change with a contact-detection rewrite, so it is not a clean scheduler-only measurement.

---

## The threads were mostly standing around, waiting for work

In SimuCell3D the bulk of the work is computing forces, contacts, and time steps for the cells in a 3D tissue. Under the tracing profiler the idle time concentrated in contact detection, the phase that dominates the runtime. The imbalance never crashed anything; it just quietly ate throughput as the tissue grew more complex.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-1.svg" alt="Thread imbalance versus tissue size on a log scale: the Static v1 baseline (dashed red), out to 12,851 cells, stays high while both adaptive runs sit consistently lower.">
  <figcaption>Workload imbalance versus tissue size (log scale). Solid dark-blue is RUN02's adaptive scheduler, carried to 26,534 cells; faint blue is RUN01's adaptive; the red dashed line is the Static (v1) baseline, out to v1's 12,851-cell reach. The <code>thread_imbalance_pct</code> column is the workload CoV scaled to a percent, a proxy for how uneven per-cell cost is, not a direct barrier-wait measurement. Both adaptive runs stay well below the Static baseline: Static mean 16.4%, median 16.1%, max 21.2%; RUN01 adaptive mean 10.6%, median 12.2%, max 14.4%; RUN02 adaptive max 15.7%.</figcaption>
</figure>

The bottleneck lived in the OpenMP directives. `Static` mode leans entirely on fixed scheduling: three bare `#pragma omp parallel for` in `solver.cpp` with no `schedule()` clause (on GCC and Clang the default is static, equal-sized contiguous chunks), an explicit `schedule(static)` in `time_integration.cpp`, and another bare parallel loop in the contact model. Static scheduling slices the cells into equal index ranges up front - thread 0 takes cells 0-15, thread 1 takes 16-31, and so on - and never rebalances. For a tissue simulation that assumption fails on the second cell.

The costs really do diverge. Each cell is a triangulated mesh, and meshes drift apart in cost as the tissue evolves. A freshly divided child has fewer faces than its parent; a growing cell has more; a cell wedged into a contact pocket spends far longer in its per-face loops than an isolated one. Picture two cells on the same step: a fresh child with a handful of faces, and a crowded neighbor jammed against four others. Static scheduling hands them out by index, not by cost, so whichever thread drew that crowded cell on step one keeps drawing its kind for the rest of the run.

---

## A single number for "how uneven is it?"

I could not fix the imbalance without first measuring it. I needed one fast number that tells the scheduler "right now the work is lumpy, switch to a finer-grained strategy."

The **coefficient of variation** (CoV = σ/μ) measures how spread out a distribution is relative to its mean. When every cell costs the same, CoV is zero and static scheduling is optimal. As some cells grow much heavier than others, CoV climbs and dynamic scheduling starts to win. The catch: you cannot measure CoV by *running* the loop, because running it is the work you are trying to schedule. The cost has to be estimated *before* the loop starts.

So the estimator in `src/solver.cpp` is a weighted sum of structural features per cell: a `base_cost` proportional to face count; a `contact_cost` scaled by a compile-time constant for the active contact model (`0.25` for node-face springs, `0.28` for node-node coupling, `0.32` for face-face coupling); an `integration_cost` of `0.65 x base_cost` for dynamic cells; plus smaller terms for polarization, growth, and mesh quality. A `#if CONTACT_MODEL_INDEX` switch keeps exactly one contact model live in a given binary. The coefficients were fit by hand against measured per-cell timings and rounded to two decimals.

Run the two cells from a moment ago through it. The fresh child, a handful of faces and no contacts, scores a low `base_cost` and almost no `contact_cost`. The crowded neighbor, more faces and several live contacts, scores several times higher on both. CoV is just how far apart those per-cell scores spread across the whole tissue, and the estimator hands the scheduler that one number without ever running a force loop.

None of this is precise, and it does not need to be. The accurate alternative, profiling every cell's real cost each step, would cost more than the imbalance it removes. A cheap structural proxy wins as long as it stays *roughly monotonic* with real cost and runs fast. Both hold: the estimator is `O(N_cells)` and returns a single float.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-2.svg" alt="Workload CoV versus tissue size for the RUN02 adaptive run against the Static v1 baseline (out to 12,851 cells), with dashed 0.4 and 0.6 threshold lines that no curve reaches." loading="lazy">
  <figcaption>The same imbalance signal as figure 1, now in raw CoV and plotted against the scheduler's decision lines. RUN02 adaptive (solid blue, to 26,534 cells) and RUN01 adaptive (faint) against the Static (v1) baseline (red dashed, out to v1's 12,851-cell reach). No curve reaches the 0.4 or 0.6 chunk-band thresholds (dashed, from <code>calculate_optimal_chunk_size()</code>), so the adaptive chunk divisor stays at its coarsest setting throughout. RUN01 adaptive: mean 0.106, median 0.122, max 0.144; RUN02 adaptive max 0.157. Static: mean 0.164, median 0.161, max 0.212.</figcaption>
</figure>

Before the design, one property of these runs: adaptive-mode CoV peaked at 0.144 in RUN01 and 0.157 in RUN02, and never reached the chunk-band thresholds (0.4 and 0.6) built into the scheduler. The chunk divisor stayed at 4, the coarsest setting, for both experiments. The high-CoV dynamic-chunking regime the system is designed for was never exercised on either run. That makes them a strong stress test for cell count, but not for the scheduler's high-heterogeneity branch.

---

## Three scheduling modes, chosen by how lumpy the work is

*This section and the next are the mechanism behind the result. If you only want the payoff, skip to "The numbers."*

`static` hands each thread a fixed pile up front: fast, but one thread can get stuck with all the slow cells. `dynamic` gives everyone a shared queue they pull from as they finish: no idle threads, but a small per-grab cost. `guided` is the middle ground, big grabs first, shrinking toward the tail. If you have ever tuned dynamic batching or a work-stealing pool, this is the same trade-off: granularity against coordination overhead.

`calculate_optimal_chunk_size()` turns the CoV directly into a granularity via a divisor: `4` for CoV ≤ 0.4 (mild imbalance, coarse chunks), `10` for 0.4 < CoV ≤ 0.6 (moderate), `20` for CoV > 0.6 (high imbalance, fine chunks). Then `chunk = max(1, min(num_cells / (num_threads x divisor), 100))`:

```cpp
// CoV -> chunk granularity (src/solver.cpp, paraphrased)
int divisor = cov <= 0.4 ? 4      // mild imbalance -> coarse chunks
            : cov <= 0.6 ? 10     // moderate
            :              20;    // high imbalance -> fine chunks
// below 100 cells, always uses divisor=4 (hardcoded fast path)
int chunk = std::max(1, std::min(num_cells / (num_threads * divisor), 100));
```

A bigger divisor means smaller chunks, and a smaller chunk is just a smaller batch: more trips to the shared queue, finer load balancing.

The chunk-size formula is only half of mode selection. In adaptive mode the actual path is a two-component process:

- First, `lookup_benchmark_mode` consults a 13-entry static table keyed on cell-count range alone, not on thread count or CoV, mapping each range to a fixed schedule mode. The table entries carry hardcoded rationale strings from earlier profiling.
- Second, `multi_factor_heuristic` runs independently, using CoV as its primary input: CoV > 0.6 → dynamic; CoV < 0.15 and tasks_per_thread ≥ 4 → static; 512 ≤ cells ≤ 4096 at moderate CoV → guided.

**When the two paths disagree, the benchmark table always wins.** The heuristic's suggestion is logged in verbose output but never triggered. Building a heuristic and then hardcoding it to lose to a table looks backwards, but I trust measured speedup data over a computed estimate.

The less obvious point is that *different loop categories have different workload shapes*, so they should not share one schedule. `initialize_per_loop_schedules()` sets four fixed structural assignments:

- **Contact detection** → `omp_sched_dynamic`, chunk = max(1, base_chunk). The most irregular phase; cost varies with mesh density and contact geometry.
- **Time integration** → `omp_sched_guided`, chunk = max(1, base_chunk x 2). More uniform per-cell cost; guided's shrinking grabs capture most of the benefit without dynamic's per-grab overhead.
- **Mesh updates (face-type classification)** → `omp_sched_static`, chunk = 0 (equal distribution). The one category where per-cell cost genuinely is uniform, and static maximizes cache locality.
- **Cell division** → `omp_sched_dynamic`, chunk = max(1, base_chunk / 2). The rarest and most variable phase; finer-grained dynamic avoids stragglers on the few iterations it fires.

Most hot `#pragma omp parallel for` loops in the current fork carry `schedule(runtime)`, which is what enables this late binding. Two bare loops without a schedule clause remain in `cell_divider.cpp:27` and `poisson_sampling.cpp:142`; neither is wired into the adaptive machinery yet.

The last piece is *adaptation over time*. Every 50 iterations (`COV_UPDATE_INTERVAL = 50`), the solver recalculates the workload CoV and then runs phase-aware schedule adaptation. Non-adaptive modes can update their chunk size when CoV and the implied chunk size both shift enough; adaptive mode keeps CoV fresh but only changes the global schedule on phase transitions, which, as the next section shows, almost never happened on this run, leaving the per-loop schedules to do the real work.

---

## A three-phase design, two phases exercised

The per-loop schedules above decide *which loop* gets which strategy. Phases are the other axis. They decide *when* in the tissue's life to shift the global default, as the simulation moves from a few scattered cells to a settled sheet. The scheduler tracks three:

- **INITIALIZATION:** `dynamic`. Thread count exceeds task count, so any fixed assignment starves threads.
- **GROWTH:** triggered when the recent division rate exceeds `0.01`, where the rate is $\text{recent divisions} / (N_{\text{cells}} \times 50)$. Then `dynamic` if CoV > 0.4, else `guided`. Cell counts and costs are changing fast.
- **HOMEOSTASIS:** `static` if cell count > 1,000, else `guided`. The tissue has settled and costs are stable enough that static's cache locality pays off at scale.

In the maximum-cell fast-growth run, `grep GROWTH` against the adaptive performance-diagnostics log returns zero matches. GROWTH never fired in RUN01, and RUN02 behaved the same way out to 26,534 cells.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-3.svg" alt="Scheduler phase plot for both runs: INITIALIZATION through iteration 4,100, then HOMEOSTASIS out to 19,958 cells (RUN01) and 26,534 (RUN02), with GROWTH never appearing." loading="lazy">
  <figcaption>Scheduler phase decisions for both adaptive runs (RUN01 as circles, RUN02 as squares), iteration on a linear axis and cell count on a log axis. INITIALIZATION fires through iteration 4,100 while the tissue has fewer than 10 cells. Each run then jumps straight to HOMEOSTASIS and stays there to 19,958 cells (RUN01) and 26,534 (RUN02). GROWTH never appears in either.</figcaption>
</figure>

This is a property of the workload, not a design flaw. The run metadata points to `parameters/parameters_128k_fast_growth.xml`, but the phase detector still never saw a recent division rate high enough to cross the GROWTH threshold. On this workload the adaptive scheduler spent 41 samples in INITIALIZATION and 167 in HOMEOSTASIS, never touching the phase the GROWTH branch was written for; RUN02 split 40 and 179 the same way.

---

## The numbers - about 2x the cells per compute budget

The headline is cell count. Under the same compute budget - same stopping point, eight cores - adaptive roughly doubled the reachable tissue size; the exact endpoints are in figure 4 below.

I treat what follows as two independent runs, not one larger one. RUN01, the 8-core run, is the headline. RUN02, on a 7-core machine and carried until the adaptive process was killed at about 98 GB of RAM, is a reproduction that happens to reach further: 26,534 adaptive cells against 12,851 for the v1 baseline run alongside it, a 2.06x ratio matching RUN01 exactly. The 2x envelope held; the one midrange dip from RUN01 did not return. Where a figure can show both, RUN02 is plotted next to RUN01; elsewhere the Static (v1) baseline is carried along as a dashed reference out to v1's 12,851-cell reach. The 26,534 is a memory-bound ceiling, not a chosen stopping point, and the two machines differ, so on the wall-clock plots read the shape, not the absolute time.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-4-wallclock-cell-growth.svg" alt="Cell count versus wall-clock time on log2-log2 axes for both runs: adaptive reaches about twice the cells of v1 at the same stopping point." loading="lazy">
  <figcaption>Cell count versus wall-clock, both axes log2 so every gridline is one doubling. RUN01 (8 cores, solid lines) reached 19,958 adaptive cells against 9,693 for v1; RUN02 (7 cores, dashed) reached 26,534 against 12,851. About 2x more cells for the same budget in each. The two runs are different machines, so read the shape, not the absolute time: on these axes adaptive growth is a near-straight line, cells proportional to t^0.88.</figcaption>
</figure>

Replotting that against **linear** time on the x-axis makes the doubling time itself legible:

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-4b-doubling-semilog.svg" alt="log2 cells versus linear wall-clock hours for both runs: the curve bends upward because each doubling takes longer, with adaptive above v1 throughout." loading="lazy">
  <figcaption>The same data, log2 cells against linear time (RUN01 solid, RUN02 dashed). A straight line would mean a constant doubling time; the curve bends because each doubling takes longer, roughly 2x the wall-clock of the one before. For adaptive in RUN01, one doubling stretches from about 3 hours at 2k-4k cells to about 14 hours at 8k-16k. In both runs the adaptive curve stays above v1's throughout, so it reaches each rung sooner; adaptive ends at 19,958 cells in RUN01 and 26,534 in RUN02.</figcaption>
</figure>

From the rate side, throughput tells it too, iterations per second as the tissue grows:

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-5.svg" alt="Iterations per second versus cell count on log-log axes for both runs: adaptive continues well past where v1 terminates." loading="lazy">
  <figcaption>Iterations per second versus cell count (log-log), both runs (RUN01 as circles, RUN02 as squares). Adaptive uses per-iteration computational diagnostics; v1 in RUN01 uses the cleaned monitor data, in RUN02 the per-iteration timing. In RUN01 v1 terminates at 9,693 cells while adaptive continues to 19,958; RUN02 repeats the pattern and carries adaptive to 26,534 against v1's 12,851.</figcaption>
</figure>

At matched cell counts the picture is cleaner than the raw endpoint IPS: adaptive is faster in every usable band after 25 cells, with one midrange dip around 250-499 cells in RUN01. RUN02 is steadier and shows no such dip.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-6-speedup-by-cell-band.svg" alt="Matched-cell speedup by cell-count band for both runs: RUN01 peaks near 2.75x with a dip near 250-499 cells, RUN02 holds a steadier 2.2 to 2.6x with no dip." loading="lazy">
  <figcaption>Matched-cell speedup, binned by cell count: median adaptive IPS divided by median v1 IPS in the same band, for RUN01 (blue) and RUN02 (grey). n is the adaptive/v1 sample count per band. RUN01 peaks at 2.75x (150-249 cells) with a dip to 1.50x at 250-499; RUN02 holds a steadier 2.2 to 2.6x and the dip does not return, while extending to a 10k-20k band RUN01 never reached. The 10-24 cell band is omitted because v1 has only one cleaned sample there in RUN01.</figcaption>
</figure>

The scaling exponent (time per iteration ~ N^α) is less clean here because the v1 side comes from monitor-derived cleaned data rather than per-iteration diagnostics. Adaptive gives α = 1.133 (R² = 0.999) for cells ≥ 10. The v1 monitor fit gives α = 1.132 but with much weaker fit quality (R² = 0.835), so I would not lean on the exponent comparison as primary evidence. RUN02, whose v1 side is per-iteration timing rather than the noisy monitor, fits more cleanly (v1 R² = 0.997), but the matched-cell speed bands are still the more defensible summary.

The physics barely moves: median pressure deviation between the two modes is 2.93% across 389 matched iterations in the cleaned biological outputs (RUN01), and 2.33% across 403 in RUN02. Reordering the work changes only how fast the simulation reaches that answer.

---

## Where the runtime actually goes

Breaking one mean iteration down by phase shows where each scheduler actually spends its time.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-7.svg" alt="Share of mean iteration time by phase: contact detection drops from 82% under Static to about 54% under Adaptive in both runs." loading="lazy">
  <figcaption>Share of mean iteration time by phase (cells > 100), RUN01 Static (v1) against adaptive from both runs. Static: contact detection 82%, polarization plus internal forces 14%, time integration 3%, mesh refinement 2%. Adaptive, RUN01: 54% / 33% / 8% / 5%. Adaptive, RUN02: 57% / 30% / 7% / 6%, a matching redistribution. RUN02 has no Static row of its own, since genuine v1.0 emits no per-phase timing.</figcaption>
</figure>

Contact detection drops from ~82% of mean iteration time to ~54% in RUN01 and ~57% in RUN02. Polarization and internal forces become the second-largest phase, but only because contact detection got faster, not because that work grew. Time integration stays a minor 3-8%, never the dominant cost.

Here is the caveat I have been deferring, stated once and in full. This run supports the matched-cell throughput comparison, but it does **not** isolate scheduler effects from the contact-detection changes. The drop from 82% to 54% is not only scheduling: the contact-detection rewrite (a uniform spatial grid, Morton sorting, and a Sweep-and-Prune broad-phase, described below) runs alongside it. The biggest likely lever is still that contact-detection work, faster data structures plus better scheduling of an inherently irregular phase, but I do not have a clean ablation between the USPG/Morton changes and the scheduler. Turning off Morton sorting and re-running the maximum-cell workload is the one measurement this section is missing.

---

## Where the rest of the speedup came from

**Faster contact detection.** Contact detection is both the most irregular phase and the most expensive, so speeding it up multiplies with the scheduling win. Its spatial-lookup containers (an unbounded uniform grid, "USPG") switched from `std::forward_list` to `std::vector`, for better cache behavior and fewer pointer chases. Morton sorting of faces before USPG insertion was added separately: faces near each other in space land near each other in memory, and the cache stops thrashing. The fork also adds a Sweep-and-Prune (SAP) broad-phase, a cheap first pass that discards pairs of cells that cannot possibly touch before any exact contact test, and an `adaptive` contact-detection setting that switches to SAP above `ADAPTIVE_SAP_CELL_THRESHOLD = 500`. SAP sorts each cell's bounding-box projections along the coordinate axes and reports interval overlaps. Like any broad-phase it is conservative: it can flag pairs that do not actually touch, but it never prunes a pair that does, and the exact narrow-phase test still runs on the survivors. One caveat: `contact_detection_algorithm` still defaults to `uspg`.

**Better CI and memory-safety tooling.** Static's (`v1.0`) CI ran one release build and `ctest -C Release`. The fork adds Debug builds, AddressSanitizer and UndefinedBehaviorSanitizer, a clang-format check, and latency profiling. The sanitizers earned their place immediately: they caught a heap-use-after-free in `local_mesh_refiner::split_edge`, a reference left dangling after a vector reallocation, that had survived careful manual review.

**A correctness sweep.** Alongside the performance work: a division-by-zero guard in `mat33::inverse`, NaN suppression in `vec3::angle`, a fix for a `cell_lst.size()` data race in the parallel cell-division loop (`cell_divider.cpp`), and null checks in `parameter_reader`. The parallel division bug is the kind that *only* surfaces under the heavier thread utilization the new schedules produce, reason enough to fix it before trusting any benchmark.

---

## What's still open

A few gaps remain. I will come back to them when my day job leaves room:

- **Thread Sanitizer is not in CI yet.** Only ASan and UBSan run. The cell-division parallel section still produces enough benign-looking races that TSan is noisy, and that noise needs triaging before it can gate the build.
- **The Python wrapper has not caught up.** The new `--schedule=` and `--diagnostics-csv=` CLI flags exist on the C++ binary but are not exposed through the pybind11 wrapper. `simucell3d_wrapper` only forwards simulation parameters, cell list, thread count, and verbosity, so Python users cannot reach the new knobs. Tracked for the next release.
- **`assert()` in hot paths.** Over 300 `assert()` calls still live in production `src/` (all runtime `assert()`, none `static_assert`). Several have been converted to proper runtime checks in the critical paths; the rest are a slower migration.
- **The high-CoV machinery is unexercised on the max-cell workload.** The 0.4/0.6 chunk-band thresholds and the GROWTH phase detector are implemented and confirmed in the source, but neither fired even in the fast-growth runs that reached 19,958 and then 26,534 cells. A workload with sharper per-cell cost heterogeneity is the right way to exercise those branches.

---

## What this project taught me

The most consequential scheduling parameter turned out not to be the integrator or the time step, but the order in which threads pick up work. The right choice costs almost nothing at runtime; the wrong one quietly costs a large fraction of throughput, because CPU utilization stays pinned at 100% even while thread utilization tanks. `top` tells you nothing useful here. A tracing profiler tells you everything.

Instrument before optimizing. The five-minute version of this work was to flip `schedule(static)` to `schedule(dynamic)` and ship it. Building the full measurement stack took far longer, but it surfaced the result that mattered most: even on the fast-growth run, CoV never crossed its thresholds and GROWTH never fired, so most of the adaptive machinery went unexercised. The same instrumentation showed contact detection to be the dominant phase, which makes algorithmic work on it a bigger lever in these numbers than the scheduler itself.

Fix correctness before trusting a benchmark. The sanitizers found a memory bug that had survived manual review, and the parallel-division race only surfaces under the heavier thread utilization the new schedules cause. Caught in that order, neither got the chance to quietly contaminate the numbers.

If you work with high-heterogeneity tissue - sharp per-cell cost differences, not just many cells - that is exactly the regime this scheduler was built for and never got to prove on this run. The fork is below and the `--schedule=` flags are documented. I would like to see the high-CoV branch put under real load.

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

[Browse the fork on GitHub](https://github.com/nilesh-patil/simucell3d){: .btn-soft .btn-soft--primary} [Read the project page](/portfolio/simucell3d/){: .btn-soft} [Original paper (Nature Comp. Sci. 2024)](https://www.nature.com/articles/s43588-024-00620-9){: .btn-soft}

- **Code**: [GitHub repo](https://github.com/nilesh-patil/simucell3d) (tag `v2.0`, branch `main`)
- **Reference**: Runser et al., [SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9), *Nature Computational Science* (2024)
- **Project page**: [SimuCell3D on Side Projects](/portfolio/simucell3d/)
