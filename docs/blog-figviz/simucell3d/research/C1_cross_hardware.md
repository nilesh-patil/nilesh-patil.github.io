# C1 — Honestly visualizing benchmark comparisons across different hardware

Scope: the NEW run (`parallel_benchmark_20260621_055502`) is **7 cores @ 3.10 GHz**, the LIVE-post run (`parallel_benchmark_20260124_091702`) is **8 cores @ 2.80 GHz**. Both reproduce the same qualitative result (adaptive ≈ 2.06x v1 at matched wall-clock). This note maps current data-viz best practice to the 7 SimuCell3D figures, with the heaviest focus on **figs 4, 4b, 5, 6** because those carry absolute or ratio axes that behave very differently under a hardware change.

---

## 0. The one principle that drives everything

**A within-machine ratio is portable across hardware; an absolute axis is not.**

The matched-cell speedup (adaptive IPS / v1 IPS, measured on the *same* machine in the *same* run) is dimensionless. The clock frequency and core count appear in both numerator and denominator and cancel, so the ratio reflects the *structural* parallelism change, not the silicon. Speedup is "inherently normalized and tends to be more comparable across different hardware platforms than raw (absolute) throughput numbers" and "remains a more portable metric than absolute throughput" because it reflects "the structural parallelism of an algorithm" ([JMU OpenCSF — Limits of Parallelism and Scaling](https://w3.cs.jmu.edu/kirkpams/OpenCSF/Books/csf/html/Scaling.html); [KTH PDC — strong and weak scaling](https://www.kth.se/blogs/pdc/2018/11/scalability-strong-and-weak-scaling/)).

By contrast, absolute throughput "is inherently tied to the specific capabilities of the hardware it is measured on" and "raw throughput numbers from different hardware generations are not directly comparable without accounting for the hardware's characteristics" ([JMU OpenCSF](https://w3.cs.jmu.edu/kirkpams/OpenCSF/Books/csf/html/Scaling.html)). For benchmarks under a wall-clock stopping criterion specifically, "running algorithms on different machines means they receive unequal computational resources, making direct comparison invalid," and naive cross-machine comparison "increases the probability of Type I error" ([Vermetten et al., *On the Fair Comparison of Optimization Algorithms in Different Machines*, arXiv:2305.07345](https://arxiv.org/abs/2305.07345)).

**Consequence for this post:** the *cross-hardware* story must be carried by the **ratio figure (fig 6)**, not by the absolute-axis figures (4, 4b, 5). The two runs corroborate each other only on the ratio. Their wall-clock / IPS curves must not be overlaid as if on one axis.

---

## 1. Figure-by-figure recommendations

### Fig 6 — matched-cell speedup by cell band  *(THE cross-hardware figure)*
This is the only figure whose y-axis is hardware-invariant, so it is the right place to *prove* the result travels.

- **DO add the new 7-core run as a second series (overlay is correct here).** Two-to-three series with a precise crossover read favors an overlay over small multiples ([Datawrapper — small multiple line charts](https://www.datawrapper.de/blog/what-to-consider-when-creating-small-multiple-line-charts); [Wilke, *Fundamentals of Data Visualization* ch.21](https://clauswilke.com/dataviz/multi-panel-figures.html)). Plot band-wise median-adaptive-IPS / median-v1-IPS for **both** runs on the same dimensionless y-axis. If the 8-core@2.8GHz line and the 7-core@3.1GHz line both hover ~2.0–2.75x, that visual coincidence *is* the cross-hardware argument: same lever, different silicon, same ~2x.
- **Label each series with its hardware**, e.g. "8-core Xeon @2.8 GHz (Jan)" and "7-core Xeon @3.1 GHz (Jun, to 26.5k cells)". This is the SPEC full-disclosure principle applied at series granularity: "if anything affects performance or is required to duplicate the results, it must be described" ([SPECpower_ssj2008 Run & Reporting Rules](https://www.spec.org/power/docs/specpower_ssj2008-run_reporting_rules/)).
- **When collapsing bands to one headline number, use the geometric mean of the per-band ratios, not the arithmetic mean.** The geometric mean is the correct aggregator for normalized/ratio performance across bins ([Fiveable — Performance metrics & benchmarking](https://fiveable.me/introduction-computer-architecture/unit-8/performance-metrics-benchmarking/study-guide/ofBFIiAG6IRRPXrE)).
- **Keep the honest exclusions visible in the caption:** the 10–24 band omitted (one v1 sample), and for the new run the v1 *solo tail* excluded because v1 had freed cores to itself after adaptive's OOM (an "unequal computational resources" violation — exactly the fairness trap in [arXiv:2305.07345](https://arxiv.org/abs/2305.07345)).

### Fig 4 — cell count vs wall-clock (log2–log2)  *(absolute axis — keep single-machine)*
- **Do NOT overlay the 7-core run's absolute wall-clock curve on the 8-core chart.** Wall-clock at a given cell count shifts for *both* adaptive and v1 when the clock goes 2.8→3.1 GHz and cores 8→7, so a four-curve overlay would conflate a scheduling effect with a clock/core effect. Keep this figure sourced from the single LIVE 8-core run, where adaptive-vs-v1 is a same-machine comparison and therefore valid.
- **If a second machine must appear, use small multiples (facet by hardware), not overlay**, with *fixed identical axes* across panels so the eye reads "what changed in the data, not the chart" ([Tufte via Wilke ch.21](https://clauswilke.com/dataviz/multi-panel-figures.html); small multiples are favored once you exceed 2–3 lines / risk overplotting, [Forum One](https://www.forumone.com/insights/blog/good-data-visualization-practice-small-multiples/)). Two panels: "8-core @2.8 GHz" and "7-core @3.1 GHz", each showing its own adaptive-vs-v1 pair.
- **Annotate the new run's OOM termination if it is shown at all.** Its adaptive endpoint (26,534 cells) is memory-bound/OOM-killed, *not* a clean matched stopping point. Mark the terminal point with an explicit symbol + note rather than letting it read as a normal stop. This is the honesty/provenance requirement: be transparent about how the run ended ([Kellogg — honest data viz](https://insight.kellogg.northwestern.edu/article/data-visualization-honesty-infographics); [SPEC disclosure](https://www.spec.org/power/docs/specpower_ssj2008-run_reporting_rules/)).
- **Keep the fitted exponent label on the line** (`cells ∝ t^0.88`). Reading the slope directly off log-log axes is exactly what they are for: `log Y = log k + n log X`, slope = exponent ([Statistics By Jim — log-log plots](https://statisticsbyjim.com/regression/log-log-plots/) via search; [Pomona physics — power-law fitting](http://www.physics.pomona.edu/sixideas/old/labs/LRM/LR05.pdf)). Add the fit quality (R²) next to it.

### Fig 4b — log2 cells vs linear wall-hours (doubling time)  *(absolute axis — keep single-machine)*
- **Same rule as fig 4: single-machine only.** Doubling *time* in hours is an absolute duration; it compresses ~10% just from 2.8→3.1 GHz before any scheduling effect. Do not put the 7-core curve on the same hours axis.
- If you want to make the doubling-time *story* hardware-robust, express it as a **ratio** (adaptive doubling time / v1 doubling time per rung) — that ratio is portable and could be checked against the new run, whereas the raw "3h → 15h" numbers are 8-core@2.8GHz-specific. State the hardware in the caption regardless.

### Fig 5 — iterations/sec vs cell count (log-log)  *(absolute axis — keep single-machine)*
- **IPS is the most hardware-sensitive axis of all** (it scales ~linearly with clock and with usable cores), so this is the figure most likely to mislead if the two runs are mixed. Keep it sourced from the single LIVE run. The *shape* (where v1 terminates vs adaptive continuing) is the point, and that point is intra-run.
- **Carry the DATA TRAP forward in methods, not on the chart:** v1 IPS must be derived from `computation_time_(hh:mm:ss)` deltas, not the degenerate `iter_per_sec` column. Note in the caption that v1's curve is monitor-derived (coarser) than adaptive's per-iteration diagnostics — the post already flags this for the exponent fit and that disclosure should stay.
- **Label the fitted α with its R²** and keep the existing honesty that the v1 fit is weak (R²=0.835): least-squares slope on log-log "is very sensitive to bin width and sample size" and "can provide biased estimates for the power-law exponent" ([arXiv:1605.06972 — limited-sample power-law fitting](https://arxiv.org/pdf/1605.06972)). The post's choice to demote the exponent comparison and lean on matched-cell bands is the textbook-correct call.

### Figs 1, 2, 7 — imbalance %, CoV, phase-time share  *(cannot use the new run's v1 side)*
- These compare a Static/v1 line against Adaptive. **Genuine published v1.0 in the new run emits no `cov` / `thread_imbalance_pct` / per-phase-ms**, so the Static comparison *line* cannot be sourced from the new run. Keep figs 1, 2, 7 on the LIVE 8-core run. If you cite any new-run number here, it can only be the adaptive side.
- These y-axes are already dimensionless or share-based (percent imbalance, CoV, % of iteration time), so they are *mostly* hardware-portable in principle — but without a v1 baseline in the new run there is nothing to compare against, so do not attempt a cross-run overlay. Note in fig 7's caption that the contact-detection rewrite is bundled (the existing caveat).

### Fig 3 — scheduler phase decisions vs iteration
- Phase logic (INIT → HOMEOSTASIS, GROWTH never fires) reproduces in the new run, so this is a candidate for a one-line "confirmed on the 7-core run to 26.5k cells / iter 21900" note. The axes here are iteration/cell-count, not time, so they are not distorted by the clock change. Low priority.

---

## 2. Hardware-provenance annotation: concrete pattern

Apply the SPEC full-disclosure idea at blog scale — "full disclosure of results and configuration details sufficient to independently reproduce the results" ([SPECpower Run & Reporting Rules](https://www.spec.org/power/docs/specpower_ssj2008-run_reporting_rules/)). For HPC specifically, captions should carry enough metadata to reproduce, including the hardware that was *not* the same when a result is reproduced on a new system ([Olivier et al., *HPC Benchmarking: Repeat, Replicate, Reproduce*, ACM REP'25](https://dl.acm.org/doi/10.1145/3736731.3746150)).

Concrete recommendation for every figure that touches measured timing:

- **A one-line provenance stamp in the caption (or a small on-chart badge):** `CPU: Intel Xeon, 8 cores @ 2.80 GHz · param: 128k_fast_growth · run 20260124 · single node`. For any figure that mixes runs, stamp each series/panel with its own line. Reporting *n* and run conditions in the caption is a baseline honesty requirement ([CleanChart — scientific data viz](https://www.cleanchart.app/blog/scientific-data-visualization)).
- **Never put a per-core or per-GHz "normalized" number on a chart to merge the two machines.** The naive `FLOPS = cores × clock × flops/cycle` rescaling is explicitly the rejected approach; cross-machine runtime translation is hard enough that the literature uses *conservative probabilistic* models, not a divide-by-clock, precisely because simple scaling inflates Type I error ([arXiv:2305.07345](https://arxiv.org/abs/2305.07345); [ResearchGate discussion on cross-machine efficiency comparison](https://www.researchgate.net/post/How-can-you-make-a-fair-comparison-of-the-computational-efficiency-of-different-models-while-using-different-computers-with-different-specs)). If you ever show a normalized value, label it as an estimate and never as the headline.
- **State the OOM/solo-tail caveats next to the data, not only in prose:** mark adaptive's OOM endpoint on the chart; explicitly say the v1 solo tail is excluded from the ratio. "Stopping conditions" must be explicit for performance experiments where exact numerical reproduction is impossible ([HPC Benchmarking: Repeat, Replicate, Reproduce](https://dl.acm.org/doi/10.1145/3736731.3746150)).

---

## 3. Recommended cross-run integration decision (summary)

| Figure | Axis type | Portable across HW? | Recommendation |
|---|---|---|---|
| 6 (speedup by band) | ratio (dimensionless) | **Yes** | **Overlay both runs** as the cross-hardware proof; geo-mean to aggregate; label HW per series; exclude v1 solo tail |
| 4 (cells vs wall-clock) | absolute time | No | Single LIVE run only; if 2nd machine shown, **small multiples**, fixed axes; mark OOM |
| 4b (doubling time hrs) | absolute time | No | Single LIVE run only; portable version is the doubling-time *ratio* |
| 5 (IPS vs cells) | absolute rate | No (most sensitive) | Single LIVE run only; keep v1-monitor-derived disclosure |
| 1 (imbalance %) | dimensionless | Partly | LIVE run only — new run has no v1 imbalance signal |
| 2 (CoV) | dimensionless | Partly | LIVE run only — new run has no v1 CoV signal |
| 3 (phase decisions) | iteration/cells | Yes | Optional one-line "confirmed on 7-core run" note |
| 7 (phase time share) | share (%) | Partly | LIVE run only — new run has no v1 per-phase signal |

**Net:** the new 7-core run's most defensible role is as **corroboration on fig 6** (and a confirmatory sentence on figs 3 and the headline ratio), *not* as a replacement source for the absolute-axis figures. The blog's narrative ("eight cores, same stopping point") stays anchored to the LIVE run; the new run's contribution is "and it holds on different silicon at 2.6x the cell count," carried by the dimensionless ratio.

---

## 4. House-style fit

- Matched-cell ratio over raw endpoints is already the post's stated preference and is the *same* principle that makes the result hardware-portable — reinforce it, don't dilute it.
- Honest-about-caveats: OOM termination, v1 solo-tail exclusion, bundled contact-detection rewrite, weak v1 log-log fit — all belong as on-chart/caption annotations, consistent with honest-viz guidance ([Kellogg](https://insight.kellogg.northwestern.edu/article/data-visualization-honesty-infographics)).
- Themeable inline SVG: provenance stamps and HW labels should use `currentColor`, same as the rest of the figure text, so they recolor with light/dark/sepia.

---

## Sources
- JMU OpenCSF, *Limits of Parallelism and Scaling* — speedup as a normalized, portable ratio: https://w3.cs.jmu.edu/kirkpams/OpenCSF/Books/csf/html/Scaling.html
- KTH PDC blog, *Scalability: strong and weak scaling*: https://www.kth.se/blogs/pdc/2018/11/scalability-strong-and-weak-scaling/
- Vermetten et al., *On the Fair Comparison of Optimization Algorithms in Different Machines*, arXiv:2305.07345: https://arxiv.org/abs/2305.07345
- Raasveldt et al., *Fair Benchmarking Considered Difficult* (DBTest'18): https://mytherin.github.io/papers/2018-dbtest.pdf
- Olivier et al., *HPC Benchmarking: Repeat, Replicate, Reproduce* (ACM REP'25): https://dl.acm.org/doi/10.1145/3736731.3746150
- SPECpower_ssj2008 Run and Reporting Rules (full-disclosure principle): https://www.spec.org/power/docs/specpower_ssj2008-run_reporting_rules/
- Wilke, *Fundamentals of Data Visualization*, ch.21 Multi-panel figures (small multiples, fixed axes): https://clauswilke.com/dataviz/multi-panel-figures.html
- Datawrapper, *What to consider when creating small multiple line charts*: https://www.datawrapper.de/blog/what-to-consider-when-creating-small-multiple-line-charts
- Forum One, *Good Data Visualization Practice: Small Multiples*: https://www.forumone.com/insights/blog/good-data-visualization-practice-small-multiples/
- Statistics By Jim, *Using Log-Log Plots* (slope = scaling exponent): https://statisticsbyjim.com/regression/log-log-plots/
- Pomona College physics, *Power-Law Fitting and Log-Log Graphs*: http://www.physics.pomona.edu/sixideas/old/labs/LRM/LR05.pdf
- *Effect of Limited Sample Sizes on Estimated Scaling Parameter for Power-Law Data*, arXiv:1605.06972 (slope-fit caution): https://arxiv.org/pdf/1605.06972
- Fiveable, *Performance metrics and benchmarking* (geometric mean for normalized aggregation): https://fiveable.me/introduction-computer-architecture/unit-8/performance-metrics-benchmarking/study-guide/ofBFIiAG6IRRPXrE
- CleanChart, *Scientific Data Visualization* (caption metadata, report n/conditions): https://www.cleanchart.app/blog/scientific-data-visualization
- Kellogg Insight, *4 Keys to Effective and Honest Data Visualizations*: https://insight.kellogg.northwestern.edu/article/data-visualization-honesty-infographics
- ResearchGate, *Fair comparison of computational efficiency across different computers*: https://www.researchgate.net/post/How-can-you-make-a-fair-comparison-of-the-computational-efficiency-of-different-models-while-using-different-computers-with-different-specs
