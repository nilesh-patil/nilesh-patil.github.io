---
layout: single
title: "Adaptive OpenMP scheduling in SimuCell3D for tissue mechanics simulations"
date: 2026-04-16T10:00:00-04:00
last_modified_at: 2026-04-16T10:00:00-04:00
categories: [blog]
tags: [cpp, hpc, openmp, simucell3d, computational-biology, profiling]
excerpt: "Measuring work imbalance and teaching a tissue simulator's scheduler to adapt for dynamic workloads."
header:
  overlay_image: /images/blog/simucell3d/header-light.jpg
  overlay_filter: "rgba(255, 255, 255, 0.15)"
  teaser: /images/blog/simucell3d/header-light.jpg
---

<style>
/* Scoped to this post: light hero needs a dark title instead of the theme's white */
.page__hero--overlay .page__title,
.page__hero--overlay .page__lead,
.page__hero--overlay .page__meta,
.page__hero--overlay .page__meta a { color: #13233a !important; text-shadow: none !important; }
</style>

Biology often looks like chemistry, but many of its hardest questions are mechanical: how a sheet of cells bends into a tube, how a spheroid opens into a fluid-filled vesicle, or how a growing tumor pushes against the tissue around it. These processes depend not just on genes and signaling, but on force, geometry, and material properties. Cell-based simulations give us a way to study those rules directly, but detailed 3D models have usually forced a tradeoff: simplify the shape, or simulate only a small number of cells.

[SimuCell3D](https://git.bsse.ethz.ch/iber/Publications/2024_runser_simucell3d) is an open-source C++ engine built to push past that limitation. It models each cell as a triangulated mesh and simulates tissue growth and deformation at subcellular resolution, including proliferation, extracellular matrix, fluid cavities, nuclei, and the uneven mechanical properties of polarized epithelia. It can work with spheroids, vesicles, sheets, tubes, and more irregular geometries imported from microscopy images, making it useful for inferring biomechanical parameters from realistic tissue structures.

The ETH Iber lab released v1.0 in 2024. This post is a field report from the [fork](https://github.com/nilesh-patil/simucell3d) I’ve been building on top of it. Apart from a few bug fixes in the original code, I focus on a practical performance bottleneck: load imbalance in the simulator’s OpenMP-parallel force and contact loops. I’ll walk through where the imbalance came from, how I measured it, and what it took to make the scheduler adapt so large, high-detail 3D tissue simulations become more tractable. The stakes are practical: this is a single-node, 8-core workload, and at subcellular resolution a microscopy-realistic tissue runs to tens of thousands of cells and beyond. How many you can push through a fixed compute budget decides whether you’re simulating a toy spheroid or a structure you can hold up against imaging data.

On the surface, the `Static` (v1.0) binary looked fully CPU-bound. `top` showed all eight cores pegged, utilization near 800%. A tracing profiler told a different story: on the worst steps, close to a third of that "busy" time was threads idling at a barrier while one of them finished an oversized slice.

That gap is invisible to `top` because it reports *aggregate* CPU time: 800% across eight cores looks identical whether one thread sprints while seven idle or all eight share the load evenly. A per-thread trace shows which one it is. If you train models, you already know this failure by a different name: a data-parallel step where one worker drew all the long sequences and the rest sit idle at the all-reduce barrier.

> **TL;DR.** Static OpenMP scheduling handed equal-sized index ranges to every thread regardless of per-cell cost, so whichever thread drew the heavy cells kept them. I built a workload estimator, wired it to per-loop schedule selection, and let it adapt as the tissue grows. In the maximum-cell fast-growth benchmark, adaptive processed **19,958 cells** to the v1 baseline's **9,693** by the same stopping point. At matched cell counts, median adaptive throughput is mostly **~2.0–2.6× higher, peaking at ~2.75×** from 25 cells through the 5k–10k band (with one dip to ~1.5× at 250–499 cells), using cleaned v1 monitor data from that same run. The result still bundles scheduling changes with contact-detection work, so a clean scheduler-only ablation is missing.

---

## The threads were mostly standing around, waiting for work

In SimuCell3D the bulk of work is computing forces, contacts, and time steps for cells in a 3D tissue. I put the `Static` binary under a tracing profiler; the idle time concentrated in contact detection, the phase that dominates the runtime. The simulation always completed. The imbalance silently ate throughput as the tissue grew more complex.

![Measured thread imbalance vs tissue size: Static (v1) vs Adaptive](/images/blog/simucell3d/figure-1.svg)
*Thread imbalance vs tissue size (log scale), Static (v1) vs Adaptive. The `thread_imbalance_pct` column is the workload CoV (coefficient of variation - how uneven per-cell cost is) scaled to a percent, not a direct barrier-wait measurement. Adaptive sits consistently below Static across the range - Static (v1): mean 16.4%, median 16.1%, max 21.2%; Adaptive: mean 10.6%, median 12.2%, max 14.4%.*

The bottleneck lived in the OpenMP directives: `Static` mode leans entirely on fixed scheduling - three bare `#pragma omp parallel for` in `solver.cpp` with no `schedule()` clause (on GCC/Clang, the default is static with equal-sized contiguous chunks), an explicit `schedule(static)` in `time_integration.cpp`, and another bare parallel loop in the contact model. Static scheduling slices the cells into equal index ranges up front - thread 0 takes cells 0–15, thread 1 takes 16–31, and so on - and never rebalances. For a tissue simulation that assumption fails on the second cell.

And the costs really do diverge. Each cell is a triangulated mesh, and meshes drift apart in cost. A freshly divided child has fewer faces than its parent; a growing cell has more; a cell wedged in heavy contact spends far longer in its per-face loops than an isolated one. Static scheduling hands out cells by index, not by cost, so whichever thread happened to draw the heavy cells on step one keeps drawing them for the rest of the run.

---

## A single number for "how uneven is it?"

I couldn't fix the imbalance without first measuring it - I needed one fast number that tells the scheduler "right now the work is lumpy, switch to a finer-grained strategy."

On this workload, adaptive-mode CoV peaked at 0.144 and never reached the 0.4 or 0.6 chunk-band thresholds built into the scheduler. The chunk divisor stayed at 4 - the coarsest setting - for the entire experiment. The high-CoV dynamic-chunking regime the system is designed for was never exercised on this run. This makes the run a useful stress test for cell count, but not for the scheduler's high-heterogeneity branch.

**Coefficient of variation** (CoV = σ/μ) measures how spread out a distribution is relative to its mean. When every cell costs the same, CoV is zero and static scheduling is optimal. As some cells grow much heavier than others, CoV climbs and dynamic scheduling starts to win. But you can't measure CoV by *running* the loop - that's the work you're trying to schedule in the first place. The cost has to be estimated *before* the loop starts. The estimator in `src/solver.cpp` is a weighted sum of structural features per cell: a `base_cost` proportional to face count; a `contact_cost` scaled by a compile-time constant for the active contact model (`0.25` for node–face springs, `0.28` for node–node coupling, `0.32` for face–face coupling); an `integration_cost` of `0.65 × base_cost` for dynamic cells; plus smaller terms for polarization, growth, and mesh quality. A `#if CONTACT_MODEL_INDEX` switch keeps exactly one contact model live in a given binary.

The coefficients were fit by hand against measured per-cell timings and rounded to two decimals. None of this is precise, and it doesn't need to be. The accurate alternative - profiling every cell's real cost each step - would cost more than the imbalance it's trying to remove. A cheap structural proxy wins as long as it stays *roughly monotonic* with real cost and runs fast. Both hold: the estimator is **O(N_cells)** and returns a single float.

![Workload CoV vs tissue size: Static (v1) vs Adaptive with 0.4 and 0.6 reference lines](/images/blog/simucell3d/figure-2.svg)
*Workload CoV vs tissue size, Static (v1) vs Adaptive. Dashed lines mark the 0.4 and 0.6 chunk-size-band thresholds built into `calculate_optimal_chunk_size()`. Adaptive: mean 0.106, median 0.122, max 0.144. Static (v1): mean 0.164, median 0.161, max 0.212. Neither mode reached either threshold band, so the adaptive chunk divisor stayed at its coarsest setting throughout.*

---

## Three scheduling modes, chosen by how lumpy the work is

`static` hands each thread a fixed pile up front (fast, but one thread can get stuck with all the slow cells); `dynamic` gives everyone a shared queue they pull from as they finish (no idle threads, but a small per-grab cost); `guided` is the middle ground - big grabs first, shrinking toward the tail. If you've ever tuned dynamic batching or a work-stealing pool, this is the same trade-off: granularity versus coordination overhead.

The function `calculate_optimal_chunk_size()` turns the CoV directly into a choice of granularity via a divisor: `4` for CoV ≤ 0.4 (mild imbalance, coarse chunks), `10` for 0.4 < CoV ≤ 0.6 (moderate), `20` for CoV > 0.6 (high imbalance, fine chunks). Then `chunk = max(1, min(num_cells / (num_threads × divisor), 100))`:

```cpp
// CoV → chunk granularity (src/solver.cpp, paraphrased)
int divisor = cov <= 0.4 ? 4      // mild imbalance → coarse chunks
            : cov <= 0.6 ? 10     // moderate
            :              20;    // high imbalance → fine chunks
// below 100 cells, always uses divisor=4 (hardcoded fast path)
int chunk = std::max(1, std::min(num_cells / (num_threads * divisor), 100));
```

The chunk-size formula is only half of mode selection. The actual path the code takes in adaptive mode is a two-component process:
- First, `lookup_benchmark_mode` consults a 13-entry static table keyed on cell-count range alone - not on thread count or CoV - mapping each range to a fixed schedule mode. The lookup table entries carry hardcoded rationale strings from earlier profiling.
- Second, `multi_factor_heuristic` runs independently, using CoV as its primary input: CoV > 0.6 → dynamic; CoV < 0.15 and tasks_per_thread ≥ 4 → static; 512 ≤ cells ≤ 4096 at moderate CoV → guided.

**When the two paths disagree, the benchmark table always wins.** The heuristic's suggestion is logged in verbose output but not triggered. It looks backwards to build a heuristic and then hardcode it to lose to a table, but I trust measured speedup data over a computed estimate, so the table wins.

The less obvious insight is that *different loop categories have different workload shapes*, so they should not share one schedule. `initialize_per_loop_schedules()` sets four fixed structural assignments:

- **Contact detection** → `omp_sched_dynamic`, chunk = max(1, base_chunk). Most irregular; costs vary with mesh density and contact geometry.
- **Time integration** → `omp_sched_guided`, chunk = max(1, base_chunk × 2). More uniform per-cell cost; guided's shrinking-grab schedule captures most benefit without dynamic's per-grab overhead.
- **Mesh updates (face-type classification)** → `omp_sched_static`, chunk = 0 (equal distribution). The one loop category where cost per cell genuinely is uniform - static maximises cache locality here.
- **Cell division** → `omp_sched_dynamic`, chunk = max(1, base_chunk / 2). Rarest and most variable phase; finer-grained dynamic avoids stragglers on the few iterations it fires.

Most hot `#pragma omp parallel for` loops in the current fork carry `schedule(runtime)`, enabling this late-binding approach. Two bare loops without a schedule clause remain in `cell_divider.cpp:27` and `poisson_sampling.cpp:142`; these are not yet wired into the adaptive machinery.

The last piece is *adaptation over time*. Every 50 iterations (`COV_UPDATE_INTERVAL = 50`), the solver recalculates the workload CoV and then runs phase-aware schedule adaptation. Non-adaptive modes can update their chunk size when CoV and the implied chunk size both shift enough; adaptive mode keeps CoV fresh but only changes the global schedule on phase transitions. The per-loop schedules are still the main mechanism in this benchmark.

---

## A three-phase design, two phases exercised

In the maximum-cell fast-growth run, `grep GROWTH` against the adaptive performance-diagnostics log returns zero matches. GROWTH never fired, even though this is the run that reached the largest tissue.

The scheduler tracks three simulation phases:

- **INITIALIZATION** (fewer than 10 cells): `dynamic`. Thread count exceeds task count; any fixed assignment starves threads.
- **GROWTH** (division_rate > 0.01, where division_rate = recent_division_count\_ / (num_cells × 50)): `dynamic` if CoV > 0.4, else `guided`. Cell counts and costs are changing fast.
- **HOMEOSTASIS** (all other cases): `static` if cell count > 1,000, else `guided`. The tissue has settled; costs are stable enough that static's cache locality pays off at scale.

![Semi-log scheduler phase plot over iteration and cell count](/images/blog/simucell3d/figure-3.svg)
*Semi-log view of scheduler phase decisions across the maximum-cell fast-growth run (adaptive mode): the iteration axis is linear, the cell-count axis is log. INITIALIZATION fires through iteration 4,100 while the tissue has fewer than 10 cells. Then the tissue transitions directly to HOMEOSTASIS and stays there for the remainder - growing from 11 to 19,958 cells. GROWTH never appears in the adaptive diagnostics.*

This is a property of the workload, not a design flaw. The run metadata points to `parameters/parameters_128k_fast_growth.xml`, but the phase detector still did not see a high enough recent division rate to cross its GROWTH threshold. On this workload, the adaptive scheduler spent 41 samples in INITIALIZATION and 167 samples in HOMEOSTASIS, never touching the phase the GROWTH branch was written for.

---

## The numbers

The headline is cell count. Under the same compute budget - same stopping point, eight cores - the maximum-cell run reached roughly twice the tissue in adaptive mode: about 20k cells versus 10k in v1.0.

![Cell growth vs wall-clock time on log₂–log₂ axes, Static (v1) vs Adaptive](/images/blog/simucell3d/figure-4-wallclock-cell-growth.svg)
*Cell count at the same stopping point on 8 cores, with both axes log₂ - every gridline is one doubling. The comparison log ends at 19,912 adaptive cells versus 9,693 v1 (the adaptive diagnostic endpoint records 19,958; the checkpoint near the stop, 19,614 vs 9,399) - about **2×** more cells for the same compute budget. On these axes the growth is a near-straight power law (cells ∝ t^0.88 for adaptive), so each tissue doubling costs roughly **2× more wall-clock time** than the last - adaptive just climbs the ladder faster.*

The same run with **linear** time on the x-axis makes the doubling time itself legible:

![Doubling time growing: log₂ cells vs linear wall-clock hours, Static (v1) vs Adaptive](/images/blog/simucell3d/figure-4b-doubling-semilog.svg)
*A semi-log₂ view of the same data - log₂ cells against linear time. A straight line would mean a constant doubling time; the curve bends because each doubling takes longer. For adaptive, one doubling stretches from ~3 hours at 2k–4k cells to ~15 hours at 8k–16k. Its curve stays above v1's throughout, so it reaches each rung sooner.*

Throughput tells the same story from the rate side - iterations per second as the tissue grows:

![Throughput vs tissue size (log-log) for v1 and adaptive across the full run](/images/blog/simucell3d/figure-5.svg)
*Iterations per second vs cell count (log-log), from the maximum-cell run. Adaptive uses per-iteration computational diagnostics; v1 uses the cleaned monitor data from the same benchmark. v1 data terminates at 9,693 cells; adaptive continues to 19,958.*

At matched cell counts the picture is clearer than the raw endpoint IPS: adaptive is faster in every usable band after 25 cells, with one midrange dip around 250–499 cells.

![Matched-cell speedup by cell-count band](/images/blog/simucell3d/figure-6-speedup-by-cell-band.svg)
*Matched-cell speedup from the maximum-cell run, binned by cell count. Values are median adaptive IPS divided by median v1 IPS in the same band. The 10–24 cell band is omitted because v1 has only one cleaned sample there.*

The scaling exponent (time per iteration ~ N^α) is less clean in this run because the v1 side comes from monitor-derived cleaned data rather than per-iteration diagnostics. Adaptive gives α = 1.133 (R² = 0.999) for cells ≥ 10. The v1 monitor fit gives α = 1.132 but with much weaker fit quality (R² = 0.835), so I would not use the exponent comparison as primary evidence here. The matched-cell speed bands are the more defensible summary.

Reassuringly, the physics barely moves: median pressure deviation between the two modes is 2.93% across 389 matched iterations in the cleaned biological outputs. Reordering the work doesn't change the answer, only how fast you get it.

---

## Where the runtime actually goes

Breaking one mean iteration down by phase shows where each scheduler actually spends its time.

![Where each scheduler spends its iteration: Static (v1) vs Adaptive](/images/blog/simucell3d/figure-7.svg)
*Share of mean iteration time by phase (cells > 100), Static (v1) vs Adaptive. Static (v1): contact detection 82%, polarization + internal forces 14%, time integration 3%, mesh refinement 2%. Adaptive: contact detection 54%, polarization + internal forces 33%, time integration 8%, mesh refinement 5%.*

Contact detection drops from ~82% of mean iteration time to ~54%. That isn't only scheduling: the contact-detection rewrite - a uniform spatial grid (USPG), Morton sorting, and a Sweep-and-Prune broad-phase, all detailed below - runs alongside it. Polarization and internal forces become the second-largest phase, but only because contact detection got faster, not because that work grew. Time integration stays a minor 3–8%, never the dominant cost.

One caveat the run can't escape: it supports the matched-cell throughput comparison, but does **not** isolate scheduler effects from the contact-detection changes. The biggest likely lever is still the contact-detection work - faster data structures plus better scheduling of an inherently irregular phase - but I do not have a clean ablation between the USPG/Morton changes and the scheduler. Turning off Morton sorting and re-running the maximum-cell workload is the measurement this section is missing.

---

## Three changes that compounded the gains

**Faster contact detection:**

Contact detection is both the most irregular phase and the most expensive, so speeding it up multiplies with the scheduling win. Its spatial-lookup containers (an unbounded uniform grid, "USPG") switched from `std::forward_list` to `std::vector` - better cache behaviour and fewer pointer chases. Morton sorting of faces before USPG insertion was added separately: faces near each other in space land near each other in memory and the cache stops thrashing. The fork also adds a Sweep-and-Prune (SAP) broad-phase path - a cheap first pass that prunes pairs of cells that can't possibly touch before any exact contact test - and an `adaptive` contact-detection setting that switches to SAP above `ADAPTIVE_SAP_CELL_THRESHOLD = 500`. SAP sorts each cell's bounding-box projections along the coordinate axes and reports interval overlaps. Like any broad-phase it's conservative - it can flag pairs that don't actually touch, but it never prunes a pair that does, so the exact narrow-phase test still runs on the survivors. Important caveat: `contact_detection_algorithm` defaults to `uspg`. The benchmark metadata records only that this run used `parameters/parameters_128k_fast_growth.xml`, and I don't have that exact XML in the current checkout - so I can't confirm whether it overrode the default. Treat SAP as implemented machinery, not an isolated contributor to these benchmark numbers.

**Better CI and memory-safety tooling:**

v1.0's CI ran one Release build and `ctest -C Release`. The fork adds Debug builds, AddressSanitizer and UndefinedBehaviorSanitizer, a clang-format check, and latency profiling. The sanitizers earned their place immediately: they caught a heap-use-after-free in `local_mesh_refiner::split_edge` - a reference left dangling after a vector reallocation - that had survived careful manual review.

**A correctness sweep:**

Alongside the performance work: a division-by-zero guard in `mat33::inverse`, NaN suppression in `vec3::angle`, a fix for a `cell_lst.size()` data race in the parallel cell-division loop (`cell_divider.cpp`), and null checks in `parameter_reader`. The parallel division bug is the kind that *only* surfaces under the heavier thread utilization the new schedules produce - reason enough to fix it before trusting any benchmark.

---

## What's still open

A few gaps remain. I'll come back to them when my day job leaves room:

- **Thread Sanitizer isn't in CI yet.** Only ASan and UBSan run. The cell-division parallel section still produces enough benign-looking races that TSan is noisy, and that noise needs triaging before it can gate the build.
- **The Python wrapper hasn't caught up.** The new `--schedule=` and `--diagnostics-csv=` CLI flags exist on the C++ binary but aren't exposed through the pybind11 wrapper - `simucell3d_wrapper` only forwards simulation parameters, cell list, thread count, and verbosity. Python users can't reach the new knobs yet. Tracked for the next release.
- **`assert()` in hot paths.** 306 `assert()` calls still live in production `src/` (all runtime `assert()`, none `static_assert`). Several have been converted to proper runtime checks in the critical paths; the rest are a slower migration.
- **The high-CoV machinery is unexercised on the max-cell workload.** The 0.4/0.6 chunk-band thresholds and the GROWTH phase detector are implemented and confirmed in the source, but neither fired even in the fast-growth run that reached 19,958 cells. A workload with sharper per-cell cost heterogeneity would be the right way to exercise those branches.

---

## What this project taught me

The most consequential scheduling parameter turned out not to be the integrator or the time step, but the order in which threads pick up work. The right choice costs almost nothing at runtime; the wrong one can quietly cost a large fraction of throughput, because CPU utilization stays pinned at 100% even while thread utilization tanks. `top` tells you nothing useful here - a tracing profiler tells you everything.

The second lesson was to instrument before optimizing. The five-minute version was to flip `schedule(static)` to `schedule(dynamic)` and ship it. Building the full measurement stack took longer, but it surfaced the result that mattered most: even on the fast-growth run, CoV never crossed its thresholds and GROWTH never fired, so most of the adaptive machinery went unexercised. The same instrumentation showed that contact detection is the dominant phase - which makes algorithmic work on it (the USPG/Morton/SAP rewrite) a bigger lever in these numbers than the scheduler itself.

The third was to fix correctness before trusting a benchmark. The sanitizers found a memory bug that had survived manual review, and the parallel-division race only surfaces under the heavier thread utilization the new schedules cause. Caught in that order, neither got the chance to quietly contaminate the numbers.

---

If you work with high-heterogeneity tissue - sharp per-cell cost differences, not just many cells - that's exactly the regime this scheduler was built for and never got to prove on this run. The fork is below and the `--schedule=` flags are documented; I'd like to see the high-CoV branch put under real load.

- **Code**: [github.com/nilesh-patil/simucell3d](https://github.com/nilesh-patil/simucell3d) (tag `v2.0`, branch `main`)
- **Reference**: Runser et al., [SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9), *Nature Computational Science* (2024)
- **Project page**: [SimuCell3D on Side Projects](/portfolio/simucell3d/)
