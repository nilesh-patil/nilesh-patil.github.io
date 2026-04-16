---
layout: single
title: "Adaptive OpenMP scheduling for tissue-mechanics simulations"
date: 2026-04-16T10:00:00+05:30
last_modified_at: 2026-11-19T10:00:00+05:30
categories: [blog]
tags: [cpp, hpc, openmp, simucell3d, computational-biology, profiling]
excerpt: "A tissue simulator's threads sat pegged at 800% while the trace put roughly a third of that time idle at a barrier. A fork that rebalanced the OpenMP schedules and rebuilt the contact grid reached about 2x the cells per compute budget, and the runs cannot say how much of that belongs to the scheduler."
measurement: "about 2x the cells per compute budget"
math: true
header:
  overlay_image: /images/blog/headers/simucell3d-adaptive-openmp-scheduling.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/simucell3d-adaptive-openmp-scheduling.jpg
---

<style>
/* Scoped to this post: light hero needs a dark title instead of the theme's white */
.page__hero--overlay .page__title,
.page__hero--overlay .page__lead,
.page__hero--overlay .page__meta,
.page__hero--overlay .page__meta a { color: #13233a !important; text-shadow: none !important; }
</style>

*Corrected, November 2026: the April version of this post said the scheduler's chunk size sat at its coarsest setting, that the workload kept the `GROWTH` phase dormant, and that the fork left the physics untouched. All three were wrong for the build measured here, and the text and captions now say what the logs and the code show. RUN02 was run in June 2026 and added then, and a closing section covers what the v2.1.0 release changed.*

## A sheet of cells decides to fold

A flat layer of identical cells can stay one layer thick. It can also buckle, pile into a stratified mass, or roll itself into a closed tube. Which of these happens is a mechanical question as much as a chemical one: each cell pushes back on its volume, pulls its surface taut under cortical tension, and sticks to its neighbors with some adhesion, and the tissue settles into whatever geometry balances those forces. So understanding how a gut tubule closes or how a spheroid hollows into a fluid-filled vesicle means solving for the shape a few thousand deformable surfaces fall into when they are squeezed against each other in three dimensions. That is hard to measure in a real embryo and harder to reason about by hand, which is why people simulate it.

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

The simulator I worked on had a plainer problem than the biology. On the reference build `top` showed all eight cores pegged near 800% utilization. A tracing profiler showed that on the worst steps a large share of that busy time, roughly a third by eye, was threads idling at a barrier while one overloaded thread finished. My fork changed how the threads pick up work and rebuilt the contact grid, and it reached about 2x the cells per compute budget.

[SimuCell3D](https://git.bsse.ethz.ch/iber/Publications/2024_runser_simucell3d) takes the literal version of that picture. Every cell is a closed triangulated surface, around 121 nodes and 238 faces on average, free to deform into whatever shape the forces demand. Each membrane carries an energy potential with four terms - internal pressure, area elasticity, surface tension, and bending stiffness - and the cells interact through a contact model that handles adhesion and volumetric exclusion. Steve Runser, Roman Vetter, and Dagmar Iber built it at ETH Zurich's D-BSSE and published it in Nature Computational Science in 2024. The paper reports a benchmark that grows a tissue from a single cell to roughly 125,000 cells in about a day of compute. At that resolution the number that tends to bind is how many cells you can reach before the compute budget runs out.

I forked SimuCell3D to push that number. Under the same wall-clock budget, my [fork](https://github.com/nilesh-patil/simucell3d) reached `19,958` cells where the v1.0 baseline stopped at `9,693`. A second run on different hardware, done in June 2026 and added to this post afterwards, repeated the pattern and, carried further, reached `26,534` cells against `12,851` for the v1 baseline beside it. Both are close to twice the tissue for the same compute budget. The fork changes the order in which threads pick up work, through an adaptive OpenMP scheduler, and that is the part this post is about. It is not the only difference from v1.0: the fork also restructures the contact grid, and its code line carries two physics fixes, one in the contact model and one in the integrator, so the two builds do not compute quite the same model. The time step stayed where it was. When I instrumented the scheduler to measure the win, most of the machinery it added had barely engaged. Its CoV thresholds never tripped and its `GROWTH` phase never fired, and the speedup that remains is shared between the new schedules and a plainer contact-detection change that shipped alongside them, which these runs cannot cleanly separate. I label the two runs **RUN01**, the headline, and **RUN02**, the reproduction, and carry both through the figures below.

---

## The threads were mostly standing around

For a while I believed the 800%, since eight pegged cores is the textbook picture of a CPU-bound job. In the trace, the worst steps had threads standing at a barrier, waiting for one overloaded thread to finish an oversized slice. I never wired up a clean per-thread idle counter, so the "about a third" is what the trace looked like and I have no measured fraction behind it. Enough threads waited, often enough, that aggregate utilization badly overstated the useful work.

That hidden idle time is invisible to `top` because it reports *aggregate* CPU time. 800% across eight cores looks identical whether one thread sprints while seven idle or all eight share the load evenly. Only a per-thread trace tells you which. Under the profiler the idle time concentrated in contact detection, the phase that dominates the runtime. The imbalance never crashed anything. It cost throughput as the tissue grew.

The bottleneck lived in the OpenMP directives. `Static` mode leans entirely on fixed scheduling: bare `#pragma omp parallel for` loops in `solver.cpp` with no `schedule()` clause (on GCC and Clang the default is static, equal-sized contiguous chunks), an explicit `schedule(static)` in `time_integration.cpp`, and another bare parallel loop in the contact model. Static scheduling slices the cells into equal index ranges up front - thread 0 takes cells 0-15, thread 1 takes 16-31, and so on - and never rebalances. Equal index ranges are equal work only while cells cost the same, and in a growing tissue they stop costing the same early.

Each cell is a triangulated mesh, and meshes drift apart in cost as the tissue evolves. A freshly divided child has fewer faces than its parent; a growing cell has more; a cell wedged into a contact pocket spends far longer in its per-face loops than an isolated one. Picture two cells on the same step: a fresh low-face child, and a crowded neighbor jammed against four others. Static scheduling hands them out by index range, blind to how much each one costs, so whichever thread drew that crowded cell on step one keeps drawing its kind for the rest of the run. I could not fix that without first measuring it, and what I needed was one fast number that says, right now, how lumpy the work is.

---

## A single number for how uneven the work is

The **coefficient of variation** (CoV = σ/μ) measures how spread out a distribution is relative to its mean. It is a single number for exactly the imbalance from the last section: one overloaded thread holding the rest at a barrier is a per-cell cost distribution with a heavy tail, and CoV says how heavy that tail is. When every cell costs the same, CoV is zero and static scheduling is optimal. As some cells grow much heavier than others, CoV climbs and dynamic scheduling starts to win. But you cannot measure CoV by running the loop, because running it is the work you are trying to schedule, so the cost has to be estimated before the loop starts.

So the estimator in `src/solver.cpp` is a weighted sum of structural features per cell: a `base_cost` proportional to face count; a `contact_cost` scaled by a compile-time constant for the active contact model (`0.25` for node-face springs, `0.28` for node-node coupling, `0.32` for face-face coupling); an `integration_cost` of `0.65 x base_cost` for dynamic cells; plus smaller terms for polarization, growth, and mesh quality. The coefficients were fit by hand against measured per-cell timings and rounded to two decimals. Run the two cells from a moment ago through it. The fresh child, a handful of faces and no contacts, scores a low `base_cost` and almost no `contact_cost`; the crowded neighbor scores several times higher on both. CoV is just how far apart those scores spread across the whole tissue.

None of this is precise. The accurate alternative, profiling every cell's real cost each step, would cost more than the imbalance it removes, so a cheap structural proxy is enough as long as it stays roughly monotonic with real cost and runs fast. The estimator is `O(N_cells)` and returns a single float.

The adaptive CoV peaked at 0.144 in RUN01 and 0.157 in RUN02, and never reached the 0.4 and 0.6 chunk-band thresholds built into the scheduler. Reaching them would have changed nothing in these runs anyway. In `adaptive` mode the chunk sizes are fixed when the solver starts and are never recomputed from CoV. Both runs started from a single cell, both logs report `Chunk size: 1`, the finest setting there is, and it held for the whole run: contact detection ran `dynamic` with chunk 1 and the guided loops ran chunk 2. Both runs stress cell count. Neither exercises the high-CoV regime the scheduler was designed for.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-2.svg" alt="Workload CoV versus tissue size for both adaptive runs and the Static baseline; neither reaches the scheduler's threshold lines." loading="lazy">
  <figcaption>Workload CoV versus tissue size (log x). Solid blue is RUN02's adaptive scheduler to 26,534 cells, faint blue RUN01's, red dashed the Static baseline ("Static (v1)" in the legend). The Static curve is measured only up to 9,693 cells, and those points were recovered from the earlier published version of this figure because the raw static-mode diagnostics were not kept. From 9,693 to 12,851 cells the Static curve is a projection, not measured data. The dashed lines at 0.4 and 0.6 are the scheduler's own chunk-band thresholds (from <code>calculate_optimal_chunk_size()</code>); no curve reaches either, and in <code>adaptive</code> mode the chunk size was fixed at 1 from solver start regardless. RUN01 adaptive: mean 0.106, median 0.122, max 0.144; RUN02 adaptive max 0.157. Static: mean 0.164, median 0.161, max 0.212.</figcaption>
</figure>

---

## Three modes, four loops, three phases, and what actually ran

The scheduler picks among the three OpenMP modes: `static` hands each thread a fixed pile up front, `dynamic` serves everyone from a shared queue as they finish, and `guided` starts with big grabs that shrink toward the tail. The whole trade is granularity against coordination overhead, and which mode wins depends on how uneven the cells are.

`calculate_optimal_chunk_size()` turns that CoV into a chunk size: higher CoV, finer chunks, so a badly skewed step gets balanced harder (the full three-band divisor table lives in `src/solver.cpp`). In `adaptive` mode, though, the periodic CoV update skips the chunk step. The per-loop chunks are derived once, from the chunk size computed when the solver is constructed, and with one cell in the tissue the under-100-cells branch returns 1. So the dial was set to its finest position before the first division and nothing turned it afterwards.

Different loop categories have different workload shapes, so they should not share one schedule. `initialize_per_loop_schedules()` sets four fixed structural assignments: **contact detection** runs `omp_sched_dynamic`, the most irregular phase, its cost varying with mesh density and contact geometry; **time integration** runs `omp_sched_guided`, more uniform per-cell cost, so guided's shrinking grabs capture most of the benefit without dynamic's per-grab overhead; **mesh updates** run `omp_sched_static`, the one category where per-cell cost genuinely is uniform and static maximizes cache locality; and **cell division** runs `omp_sched_dynamic` at chunk 1, the rarest and most variable phase. Most hot `#pragma omp parallel for` loops in the fork carry `schedule(runtime)`, which is what enables this late binding.

Phases are the other axis. They decide *when* in the tissue's life to shift the global default. The scheduler tracks three: `INITIALIZATION` (`dynamic`, the first handful of cells; both logs leave it at 10), `GROWTH` (meant to trigger when more than 1% of cells divided in the last 50 iterations; then `dynamic` if CoV > 0.4, else `guided`), and `HOMEOSTASIS` (`static` if cell count > 1,000, else `guided`, a choice the code makes only at the moment of a phase change). Every 50 iterations (`COV_UPDATE_INTERVAL = 50`) the solver recalculates the CoV and reconsiders the phase.

The phases went the same way. In the maximum-cell fast-growth run, `grep GROWTH` against the adaptive performance-diagnostics log returns zero matches. The adaptive scheduler spent 41 samples in `INITIALIZATION` and 167 in `HOMEOSTASIS`; RUN02 split 40 and 179 the same way. When I first published this I blamed the workload, and that was wrong. The v2.0 phase formula divided the recent division count by $N_{\text{cells}} \times 50$ before comparing it with `0.01`, which asks for more than half the cells to divide inside one 50-iteration window. `GROWTH` could not fire on any workload. The fix is commit `2cd2106`, listed in the 2.1.0 changelog. The tissue was growing fast enough for the corrected rule: net cell growth per 50 iterations had a median of 2.2% in both runs, above 1% in 146 of 166 windows in RUN01 and 162 of 178 in RUN02. That is net growth standing in for a division count, because the diagnostics' `divisions` column was zero in every row, a second bug fixed in the same commit.

The early exit had one more consequence. Both logs record the switch to `HOMEOSTASIS` at 10 cells with `Scheduling: guided (chunk=2)`. The static-above-1,000-cells rule is only evaluated at a phase change, and the phase never changed again, so the global schedule stayed `guided` for the rest of both runs.

<figure>
<svg viewBox="0 0 620 210" role="img" aria-labelledby="ph-t ph-d" style="width:100%;height:auto;max-width:620px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="ph-t">The scheduler's three designed phases, only two of which ran</title>
<desc id="ph-d">A left to right flow of three phase boxes. INITIALIZATION and HOMEOSTASIS are solid and were entered; GROWTH is faded and was never entered. The path taken arcs from INITIALIZATION over GROWTH straight to HOMEOSTASIS.</desc>
<defs>
<marker id="ph-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>
</defs>
<path d="M101,78 C101,30 519,30 519,78" fill="none" stroke="currentColor" stroke-width="2.4" marker-end="url(#ph-arrow)"/>
<text x="310" y="24" text-anchor="middle" font-size="12.5" font-weight="700" fill="currentColor">the path both runs took</text>
<line x1="188" y1="110" x2="222" y2="110" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.28"/>
<line x1="398" y1="110" x2="432" y2="110" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.28"/>
<rect x="16" y="78" width="170" height="64" rx="10" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="2"/>
<text x="101" y="102" text-anchor="middle" font-size="15" font-weight="700" fill="currentColor">INITIALIZATION</text>
<text x="101" y="124" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.72">schedule: dynamic</text>
<rect x="225" y="78" width="170" height="64" rx="10" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.34"/>
<text x="310" y="102" text-anchor="middle" font-size="15" font-weight="700" fill="currentColor" opacity="0.4">GROWTH</text>
<text x="310" y="124" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.4">never entered</text>
<rect x="434" y="78" width="170" height="64" rx="10" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="2"/>
<text x="519" y="102" text-anchor="middle" font-size="15" font-weight="700" fill="currentColor">HOMEOSTASIS</text>
<text x="519" y="124" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.72">schedule: guided</text>
<text x="101" y="166" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.82">41 / 40 samples</text>
<text x="310" y="166" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.4">0 samples</text>
<text x="519" y="166" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.82">167 / 179 samples</text>
<text x="310" y="184" text-anchor="middle" font-size="10.5" fill="currentColor" opacity="0.5">counts are RUN01 / RUN02</text>
</svg>
<figcaption>The scheduler defines three phases and both runs entered two. Each ran <code>INITIALIZATION</code> while the tissue was tiny, switched to <code>HOMEOSTASIS</code> at 10 cells, and stayed there on a guided schedule. <code>GROWTH</code> was unreachable in this build: a bug in the v2.0 phase formula put its effective threshold at half the cells dividing within 50 iterations, where 1% was intended.</figcaption>
</figure>

---

## About 2x the cells per compute budget

I treat what follows as two independent runs and keep their numbers separate. RUN01, the 8-core run, is the headline: **19,958** adaptive cells against **9,693** for the v1 baseline. RUN02, run in June 2026 on a 7-core machine and carried until the adaptive process was killed at about 98 GB of RAM, is a reproduction that happens to reach further: **26,534** adaptive cells against **12,851** for v1, a 2.06x ratio matching RUN01 exactly. The 26,534 is where memory ran out, and the two machines differ, so on the wall-clock plots compare the shapes and treat the absolute times as machine-specific.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-4-wallclock-cell-growth.svg" alt="Cell count versus wall-clock on log2-log2 axes for both runs: adaptive reaches about twice the cells of v1 at the same stopping point." loading="lazy">
  <figcaption>Cell count versus wall-clock, both axes log2 so every gridline is one doubling. RUN01 (8 cores, solid) reached 19,958 adaptive cells against 9,693 for v1; RUN02 (7 cores, dashed) reached 26,534 against 12,851. About 2x more cells for the same budget in each. The two runs are on different machines, so compare the shapes: adaptive growth is a near-straight line, cells proportional to t^0.88 in RUN02. Each doubling still costs more than the last: in RUN02, 3.9 hours from 2k to 4k cells and 18.7 hours from 8k to 16k.</figcaption>
</figure>

At matched cell counts adaptive is faster in every usable band after 25 cells, with one midrange dip around 250-499 cells in RUN01 that RUN02 does not repeat.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-6-speedup-by-cell-band.svg" alt="Matched-cell speedup by cell-count band: adaptive stays roughly 2 to 2.75x faster than v1, with one midrange dip in RUN01." loading="lazy">
  <figcaption>Matched-cell speedup, binned by cell count: median adaptive IPS divided by median v1 IPS in the same band, for RUN01 (blue) and RUN02 (grey). RUN01 peaks at 2.75x (150-249 cells) with a single dip to 1.50x at 250-499; RUN02 holds a steadier 2.2 to 2.6x, the dip does not return, and it extends to a 10k-20k band RUN01 never reached. The 10-24 cell band is omitted because v1 has only one cleaned sample there in RUN01.</figcaption>
</figure>

The two builds do not produce the same physics, and the outputs show it. For each iteration where both modes wrote output I took the relative gap between their mean cell pressures; the median of that gap is 2.93% across 389 matched iterations in the cleaned biological outputs (RUN01), and 2.33% across 403 in RUN02. The gap has a direction. In RUN02 above 1,000 cells the adaptive build's mean pressure is the higher of the two in 84% of matched iterations, with a signed median of +1.29%. That is a systematic offset, and the code accounts for it: the v2.0 code line already carried a closest-point fix in the contact model and a semi-implicit Euler fix in the integrator, both of which apply to this configuration and neither of which is in v1.0. RUN02 is the v2.0 tag exactly. RUN01's commit is no longer in the repository, so I cannot check it directly, but both fixes predate that run. The time step is the same in both builds, and cell counts at matched iterations agree within about 5% in RUN02, with v1 slightly ahead. So the 2x compares throughput between two builds whose models differ by a few percent in pressure, which is a weaker claim than the same answer arriving sooner.

---

## Where the 2x actually came from, and what is still open

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-7.svg" alt="Phase-share dumbbell: contact detection's share of the iteration falls from 82% under Static to about 54% under Adaptive, while other phases hold." loading="lazy">
  <figcaption>Share of mean iteration time per phase, Static to Adaptive, cells > 100, as contact / polarization plus forces / time integration / mesh. Adaptive shares are run means from the diagnostics: RUN01 54 / 33 / 8 / 5, RUN02 57 / 30 / 7 / 6. The Static shares, 82 / 13 / 3 / 2, have no surviving raw data. The static-mode diagnostics were pruned, and the figure script carries those four numbers as constants taken from the earlier published version of this chart. "Static (v1)" was the fork run in its static schedule mode, since v1.0 itself writes no diagnostics.</figcaption>
</figure>

Contact detection drops from ~82% of mean iteration time under Static, the weak end of the comparison since its raw data is gone, to ~54% in RUN01 and ~57% in RUN02. Polarization and internal forces rise to the second-largest share as contact detection shrinks, and time integration stays a minor 3 to 8% throughout. The adaptive numbers are run means and they flatten a trend: in RUN02 the contact share climbs with tissue size, 53.1% at 3k cells, 54.9% at 10k, 59.0% at 20.4k and 62.2% at 26.5k.

These runs support the matched-cell throughput comparison, but they cannot isolate the scheduler from the one contact-detection change that was live. The runs used the default detection path: the config sets no `contact_detection_algorithm`, and the code default is `uspg`. On that path the fork's real change is the spatial grid itself, whose per-voxel storage switched from a `std::forward_list` to a `std::vector` of vectors for cache locality (the source claims a 1.3-2x gain on that container). So the 82% to 54% drop bundles the new schedules with that grid change, and I cannot separate the two here.

The two fancier contact-detection features are in the fork but never ran on this workload. Morton-code sorting of faces and the Sweep-and-Prune broad-phase (the one gated on `ADAPTIVE_SAP_CELL_THRESHOLD = 500`) both live only behind the non-default `sweep_and_prune` and `adaptive` settings; under the default `uspg`, the face-face model takes its direct-grid path and calls neither. What was live in these runs was the per-loop schedules, the humbler vector-of-vectors grid, and the two physics fixes.

A few other things stayed open, and I will come back to them when my day job leaves room:

- **Thread Sanitizer is still not in CI.** Only ASan and UBSan run. In April I described the races in the cell-division parallel section as benign-looking noise. They were real. v2.1 fixed the division race, where daughters were appended to the cell list inside the parallel loop, and the races in contact resolution, where forces added to neighbouring cells' nodes were lost under contention. The changelog reports 43 ThreadSanitizer races in contact resolution before that fix and none after.
- **The Python wrapper has not caught up.** The new `--schedule=` and `--diagnostics-csv=` CLI flags exist on the C++ binary but are not exposed through the pybind11 wrapper, so `simucell3d_wrapper` cannot reach the new knobs.
- **`assert()` in hot paths.** Over 300 `assert()` calls still live in production `src/` (all runtime `assert()`, none `static_assert`). Several critical paths are converted; the rest are a slower migration.
- **Much of the built machinery was dormant in these runs.** The 0.4/0.6 chunk-band thresholds, the `GROWTH` phase detector, and the two non-default detection algorithms (Morton-sorted USPG and Sweep-and-Prune) are all in the source, and none fired on either run. I have no measurement of whether those branches help or hurt when they do fire. A workload with sharper per-cell cost heterogeneity, and a run that sets `contact_detection_algorithm` away from the default, are the right ways to load them.

---

## What v2.1 changed

*Added November 2026.* SimuCell3D v2.1.0 was tagged on 18 November 2026. The scheduler code is nearly untouched; the changes sit around it.

The default face-face contact path is now a compact face grid whose OpenMP schedules are written into the source (`dynamic` with chunk 1 over work units, `static` for the grid build), so `--schedule` no longer drives the dominant phase. Threads are pinned one per physical core by default; RUN01 and RUN02 ran unpinned. The race fixes are the two listed above plus a coupling-map race. New safeguards delete and count cells that go unstable, inside-out or self-intersecting, and the changelog states that results differ from 1.0.

No run used the tagged v2.1.0 commits. The closest, v21nb, ran before the tag on a build whose source matches v2.1.0 apart from comments and error paths; binaries were not compared. On a reconstruction of the paper's configuration it reached 30,000 cells in 4.31 hours, 100,000 in 26.98 and 125,000 in 38.77, against the paper's "about a day". Contact detection took 76% of solver time at 30k cells and 87% at 125k. Workload CoV had a median of 0.137 and a maximum of 0.236, still short of 0.4. At matched cell counts v21nb ran 3.9x faster than crash-fixed v1.0 at 1,000 to 2,000 cells, rising to 7.1x at 20,000 to 30,000, and 1.32x to 1.90x faster than a September development build. There is one run per build, on a reconstructed configuration where cells overlapped heavily in earlier long runs. The physics also differs between v1.0 and 2.x, so the v1 ratio is not like-for-like.

Every phase transition in v21nb happened below about 130 cells, and the run then stayed in `GROWTH` on a global guided schedule, chunk 2. For v2.0 at that scale, reading the code alone gives `static` above 1,000 cells, since `GROWTH` was unreachable, while the RUN01 and RUN02 logs show guided from 10 cells to the end, as described above. No v2.0 run on the large configuration exists to settle it.

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

[Fork on GitHub](https://github.com/nilesh-patil/simucell3d){: .btn-soft .btn-soft--primary} [Project page](/portfolio/simucell3d/){: .btn-soft} [SimuCell3D paper (Nature Computational Science, 2024)](https://www.nature.com/articles/s43588-024-00620-9){: .btn-soft}
