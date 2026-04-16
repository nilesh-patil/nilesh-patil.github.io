---
layout: single
title: "Adaptive OpenMP scheduling for tissue-mechanics simulations"
date: 2026-04-16T10:00:00+05:30
last_modified_at: 2026-04-16T10:00:00+05:30
categories: [blog]
tags: [cpp, hpc, openmp, simucell3d, computational-biology, profiling]
excerpt: "A tissue simulator's threads sat pegged at 800% while the trace put roughly a third of that time idle at a barrier. Measuring the imbalance and teaching the scheduler to adapt bought 2x the cells per compute budget."
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

## A sheet of cells decides to fold

A flat layer of identical cells can stay one layer thick. It can also buckle, pile into a stratified mass, or roll itself into a closed tube. The cells' chemistry alone does not decide how this process continues. The exact shape is defined by mechanics: each cell pushes back on its volume, pulls its surface taut under cortical tension, and sticks to its neighbors with some adhesion, and the tissue settles into whatever geometry balances those forces. So if you want to understand how a gut tubule closes or how a spheroid hollows into a fluid-filled vesicle, you are not tracking a gene. You are solving for the shape a few thousand deformable surfaces fall into when they are squeezed against each other in three dimensions. That is hard to measure in a real embryo and harder to reason about by hand, which is why people simulate it.

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

Here is the stake, before any of the biology pays off. On the reference simulator `top` showed all eight cores pegged near 800% utilization, the picture of a job with nothing left to give. A tracing profiler disagreed: on the worst steps a large share of that busy time, roughly a third by eye, was threads idling at a barrier while one overloaded thread finished. Teaching the scheduler to balance the work closed most of that gap and bought about 2x the cells per compute budget. The rest of this post is how.

[SimuCell3D](https://git.bsse.ethz.ch/iber/Publications/2024_runser_simucell3d) takes the literal version of that picture. Every cell is a closed triangulated surface, around 121 nodes and 238 faces on average, free to deform into whatever shape the forces demand. Each membrane carries an energy potential with four terms - internal pressure, area elasticity, surface tension, and bending stiffness - and the cells interact through a contact model that handles adhesion and volumetric exclusion. Steve Runser, Roman Vetter, and Dagmar Iber built it at ETH Zurich's D-BSSE and published it in Nature Computational Science in 2024. The paper's benchmark grows a tissue from a single cell to roughly 125,000 cells in about a day of compute. At that resolution the number that tends to bind is how many cells you can reach before the compute budget runs out.

I forked SimuCell3D to push that number. Under the same wall-clock budget, my [fork](https://github.com/nilesh-patil/simucell3d) reached `19,958` cells where the v1.0 baseline stopped at `9,693`. A second run on different hardware repeated the pattern and, carried further, reached `26,534` cells against `12,851` for the v1 baseline beside it. Both are close to twice the tissue for the same compute budget. What moved the number was the order in which threads pick up work, so I built an adaptive OpenMP scheduler to balance it. The integrator and the time step stayed where they were. Instrumenting that scheduler to measure the win is where the more informative result turned up: most of the machinery it added barely engaged. Its CoV thresholds never tripped and its `GROWTH` phase never fired, and the speedup that remains is shared between the new schedules and a plainer contact-detection change that shipped alongside them, which this run cannot cleanly separate. I label the two runs **RUN01**, the headline, and **RUN02**, the reproduction, and carry both through the figures below.

---

## The threads were mostly standing around

Come back to that 800%. `top` reports it as eight cores pegged near full utilization, the textbook picture of a CPU-bound job, and for a while I believed it. A tracing profiler told a different story. On the worst steps a large share of that "busy" time was threads standing at a barrier, waiting for one overloaded thread to finish an oversized slice. I never wired up a clean per-thread idle counter to put an exact number on it, so read the "about a third" as what the trace looked like, not a measured fraction. The shape of the problem is what mattered, and it was unmistakable: enough threads waited, often enough, that aggregate utilization badly overstated the useful work.

That hidden idle time is invisible to `top` because it reports *aggregate* CPU time. 800% across eight cores looks identical whether one thread sprints while seven idle or all eight share the load evenly. Only a per-thread trace tells you which. If you train models, you already know this failure by another name: a data-parallel step where one worker drew all the long sequences while the rest sat idle at the all-reduce barrier. Under the profiler the idle time concentrated in contact detection, the phase that dominates the runtime. The imbalance never crashed anything; it just quietly ate throughput as the tissue grew.

The bottleneck lived in the OpenMP directives. `Static` mode leans entirely on fixed scheduling: bare `#pragma omp parallel for` loops in `solver.cpp` with no `schedule()` clause (on GCC and Clang the default is static, equal-sized contiguous chunks), an explicit `schedule(static)` in `time_integration.cpp`, and another bare parallel loop in the contact model. Static scheduling slices the cells into equal index ranges up front - thread 0 takes cells 0-15, thread 1 takes 16-31, and so on - and never rebalances. For a tissue simulation that assumption fails on the second cell.

The costs really do diverge. Each cell is a triangulated mesh, and meshes drift apart in cost as the tissue evolves. A freshly divided child has fewer faces than its parent; a growing cell has more; a cell wedged into a contact pocket spends far longer in its per-face loops than an isolated one. Picture two cells on the same step: a fresh low-face child, and a crowded neighbor jammed against four others. Static scheduling hands them out by index range, blind to how much each one costs, so whichever thread drew that crowded cell on step one keeps drawing its kind for the rest of the run. I could not fix that without first measuring it, and what I needed was one fast number that says, right now, how lumpy the work is.

---

## A single number for how uneven the work is

The **coefficient of variation** (CoV = σ/μ) measures how spread out a distribution is relative to its mean. It is a single number for exactly the imbalance from the last section: the all-reduce straggler that holds everyone at the barrier is just a per-cell cost distribution with one heavy tail, and CoV is how heavy that tail is. When every cell costs the same, CoV is zero and static scheduling is optimal. As some cells grow much heavier than others, CoV climbs and dynamic scheduling starts to win. The catch: you cannot measure CoV by *running* the loop, because running it is the work you are trying to schedule. The cost has to be estimated *before* the loop starts.

So the estimator in `src/solver.cpp` is a weighted sum of structural features per cell: a `base_cost` proportional to face count; a `contact_cost` scaled by a compile-time constant for the active contact model (`0.25` for node-face springs, `0.28` for node-node coupling, `0.32` for face-face coupling); an `integration_cost` of `0.65 x base_cost` for dynamic cells; plus smaller terms for polarization, growth, and mesh quality. The coefficients were fit by hand against measured per-cell timings and rounded to two decimals. Run the two cells from a moment ago through it. The fresh child, a handful of faces and no contacts, scores a low `base_cost` and almost no `contact_cost`; the crowded neighbor scores several times higher on both. CoV is just how far apart those scores spread across the whole tissue.

None of this is precise, and it does not need to be. The accurate alternative, profiling every cell's real cost each step, would cost more than the imbalance it removes. A cheap structural proxy wins as long as it stays *roughly monotonic* with real cost and runs fast. Both hold: the estimator is `O(N_cells)` and returns a single float.

The first surprise is that the machinery barely engaged. On both runs the adaptive CoV peaked at 0.144 in RUN01 and 0.157 in RUN02, and never reached the 0.4 and 0.6 chunk-band thresholds built into the scheduler. The chunk divisor stayed at 4, its coarsest setting, for both experiments. The high-CoV dynamic-chunking regime the system is designed for was never exercised on either run, which makes both a strong stress test for cell count while leaving the scheduler's high-heterogeneity branch untested.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-2.svg" alt="Workload CoV versus tissue size for both adaptive runs and the Static baseline; neither reaches the scheduler's threshold lines." loading="lazy">
  <figcaption>Workload CoV versus tissue size (log x). Solid blue is RUN02's adaptive scheduler to 26,534 cells, faint blue RUN01's, red dashed the Static (v1) baseline to v1's 12,851-cell reach. The dashed lines at 0.4 and 0.6 are the scheduler's own chunk-band thresholds (from <code>calculate_optimal_chunk_size()</code>); no curve reaches either, so the adaptive chunk divisor stays at its coarsest setting throughout. RUN01 adaptive: mean 0.106, median 0.122, max 0.144; RUN02 adaptive max 0.157. Static: mean 0.164, median 0.161, max 0.212.</figcaption>
</figure>

---

## Three modes, four loops, three phases, and what actually ran

The scheduler picks among the three OpenMP modes: `static` hands each thread a fixed pile up front, `dynamic` serves everyone from a shared queue as they finish, and `guided` starts with big grabs that shrink toward the tail. The whole trade is granularity against coordination overhead, and which mode wins depends on how uneven the cells are.

`calculate_optimal_chunk_size()` turns that CoV into a chunk size: higher CoV, finer chunks, so a badly skewed step gets balanced harder (the full three-band divisor table lives in `src/solver.cpp`). But as figure-2 already showed, CoV never crossed the first threshold on either run, so this dial stayed at its coarsest setting the whole time. It is the first piece of built machinery that never had to work.

Different loop categories have different workload shapes, so they should not share one schedule. `initialize_per_loop_schedules()` sets four fixed structural assignments: **contact detection** runs `omp_sched_dynamic`, the most irregular phase, its cost varying with mesh density and contact geometry; **time integration** runs `omp_sched_guided`, more uniform per-cell cost, so guided's shrinking grabs capture most of the benefit without dynamic's per-grab overhead; **mesh updates** run `omp_sched_static`, the one category where per-cell cost genuinely is uniform and static maximizes cache locality; and **cell division** runs `omp_sched_dynamic` with a finer chunk, the rarest and most variable phase. Most hot `#pragma omp parallel for` loops in the fork carry `schedule(runtime)`, which is what enables this late binding.

Phases are the other axis. They decide *when* in the tissue's life to shift the global default. The scheduler tracks three: `INITIALIZATION` (`dynamic`, thread count exceeds task count), `GROWTH` (triggered when the recent division rate exceeds `0.01`, where the rate is $\text{recent divisions} / (N_{\text{cells}} \times 50)$; then `dynamic` if CoV > 0.4, else `guided`), and `HOMEOSTASIS` (`static` if cell count > 1,000, else `guided`). Every 50 iterations (`COV_UPDATE_INTERVAL = 50`) the solver recalculates the CoV and reconsiders the phase.

The phases tell the same story. In the maximum-cell fast-growth run, `grep GROWTH` against the adaptive performance-diagnostics log returns zero matches. On this workload the adaptive scheduler spent 41 samples in `INITIALIZATION` and 167 in `HOMEOSTASIS`, never touching the phase the `GROWTH` branch was written for; RUN02 split 40 and 179 the same way. The workload is what kept it dormant: the run metadata points to `parameters/parameters_128k_fast_growth.xml`, but the phase detector still never saw a recent division rate high enough to cross the threshold.

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
<text x="519" y="124" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.72">schedule: static</text>
<text x="101" y="166" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.82">41 / 40 samples</text>
<text x="310" y="166" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.4">0 samples</text>
<text x="519" y="166" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.82">167 / 179 samples</text>
<text x="310" y="184" text-anchor="middle" font-size="10.5" fill="currentColor" opacity="0.5">counts are RUN01 / RUN02</text>
</svg>
<figcaption>The scheduler defines three phases but the fast-growth workload only entered two. Both runs ran <code>INITIALIZATION</code> while the tissue was tiny, then jumped straight to <code>HOMEOSTASIS</code> and stayed; the division rate never crossed the 0.01 that triggers <code>GROWTH</code>, so the branch written for it never ran.</figcaption>
</figure>

---

## About 2x the cells per compute budget

I treat what follows as two independent runs and keep their numbers separate. RUN01, the 8-core run, is the headline: **19,958** adaptive cells against **9,693** for the v1 baseline. RUN02, on a 7-core machine and carried until the adaptive process was killed at about 98 GB of RAM, is a reproduction that happens to reach further: **26,534** adaptive cells against **12,851** for v1, a 2.06x ratio matching RUN01 exactly. The 26,534 is where memory ran out, and the two machines differ, so on the wall-clock plots compare the shapes and treat the absolute times as machine-specific.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-4-wallclock-cell-growth.svg" alt="Cell count versus wall-clock on log2-log2 axes for both runs: adaptive reaches about twice the cells of v1 at the same stopping point." loading="lazy">
  <figcaption>Cell count versus wall-clock, both axes log2 so every gridline is one doubling. RUN01 (8 cores, solid) reached 19,958 adaptive cells against 9,693 for v1; RUN02 (7 cores, dashed) reached 26,534 against 12,851. About 2x more cells for the same budget in each. The two runs are on different machines, so compare the shapes: adaptive growth is a near-straight line, cells proportional to t^0.88. Each doubling still costs more than the last, one stretching from about 3 hours at 2k-4k cells to about 14 hours at 8k-16k.</figcaption>
</figure>

At matched cell counts the picture is cleaner than the raw endpoints: adaptive is faster in every usable band after 25 cells, with one midrange dip around 250-499 cells in RUN01 that RUN02 does not repeat.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-6-speedup-by-cell-band.svg" alt="Matched-cell speedup by cell-count band: adaptive stays roughly 2 to 2.75x faster than v1, with one midrange dip in RUN01." loading="lazy">
  <figcaption>Matched-cell speedup, binned by cell count: median adaptive IPS divided by median v1 IPS in the same band, for RUN01 (blue) and RUN02 (grey). RUN01 peaks at 2.75x (150-249 cells) with a single dip to 1.50x at 250-499; RUN02 holds a steadier 2.2 to 2.6x, the dip does not return, and it extends to a 10k-20k band RUN01 never reached. The 10-24 cell band is omitted because v1 has only one cleaned sample there in RUN01.</figcaption>
</figure>

The physics barely moves. For each iteration where both modes wrote output I took the relative gap between their mean cell pressures; the median of that gap is 2.93% across 389 matched iterations in the cleaned biological outputs (RUN01), and 2.33% across 403 in RUN02. Reordering the work changes only how fast the simulation reaches that answer, and leaves the answer itself untouched.

---

## Where the 2x actually came from, and what is still open

Break one mean iteration down by phase and you can see where each scheduler spends its time.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simucell3d/figure-7.svg" alt="Phase-share dumbbell: contact detection's share of the iteration falls from 82% under Static to about 54% under Adaptive, while other phases hold." loading="lazy">
  <figcaption>Share of mean iteration time per phase, Static (v1) to Adaptive, cells > 100. Contact detection falls from 82% of the iteration under Static to 54% (RUN01) and 57% (RUN02); polarization and internal forces become the largest remaining phase because contact detection shrank around them, while that work itself held flat. Static: contact 82 / polarization plus forces 13 / time integration 3 / mesh 2. Adaptive RUN01: 54 / 33 / 8 / 5. RUN02: 57 / 30 / 7 / 6.</figcaption>
</figure>

Contact detection drops from ~82% of mean iteration time under Static to ~54% in RUN01 and ~57% in RUN02. Polarization and internal forces rise to the second-largest phase because contact detection got faster around them; the work itself held flat. Time integration stays a minor 3 to 8% throughout.

Here is the caveat, stated once and in full. This run supports the matched-cell throughput comparison, but it does **not** isolate the scheduler from the one contact-detection change that was actually live. The runs used the default detection path: the config sets no `contact_detection_algorithm`, and the code default is `uspg`. On that path the fork's real change is the spatial grid itself, whose per-voxel storage switched from a `std::forward_list` to a `std::vector` of vectors for cache locality (the source claims a 1.3-2x gain on that container). So the 82% to 54% drop bundles the new schedules with that grid change, and I cannot separate the two here.

The two fancier contact-detection features are in the fork but never ran on this workload. Morton-code sorting of faces and the Sweep-and-Prune broad-phase (the one gated on `ADAPTIVE_SAP_CELL_THRESHOLD = 500`) both live only behind the non-default `sweep_and_prune` and `adaptive` settings; under the default `uspg`, the face-face model takes its direct-grid path and calls neither. So they join the `GROWTH` phase and the CoV thresholds on the same list: implemented, confirmed in the source, and never exercised by these runs. What actually moved the number was the schedule and the humbler vector-of-vectors grid.

A few other things stayed open, and I will come back to them when my day job leaves room:

- **Thread Sanitizer is not in CI yet.** Only ASan and UBSan run. The cell-division parallel section still produces enough benign-looking races that TSan is noisy, and that noise needs triaging before it can gate the build.
- **The Python wrapper has not caught up.** The new `--schedule=` and `--diagnostics-csv=` CLI flags exist on the C++ binary but are not exposed through the pybind11 wrapper, so `simucell3d_wrapper` cannot reach the new knobs.
- **`assert()` in hot paths.** Over 300 `assert()` calls still live in production `src/` (all runtime `assert()`, none `static_assert`). Several critical paths are converted; the rest are a slower migration.
- **Much of the built machinery is dormant in the current config, so the measured throughput is a lower bound on what the scheduler can do.** The 0.4/0.6 chunk-band thresholds, the `GROWTH` phase detector, and the two non-default detection algorithms (Morton-sorted USPG and Sweep-and-Prune) are all confirmed in the source, but none fired on the fast-growth runs that reached 19,958 and then 26,534 cells. A workload with sharper per-cell cost heterogeneity, and a run that sets `contact_detection_algorithm` away from the default, are the right ways to load those branches.

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

[Fork on GitHub](https://github.com/nilesh-patil/simucell3d){: .btn-soft .btn-soft--primary} [Project page](/portfolio/simucell3d/){: .btn-soft} [SimuCell3D paper (Nature, 2024)](https://www.nature.com/articles/s43588-024-00620-9){: .btn-soft}
