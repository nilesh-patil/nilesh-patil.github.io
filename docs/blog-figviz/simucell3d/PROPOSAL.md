# PROPOSAL: folding the 26.5k-cell run into the SimuCell3D post

Decision-ready synthesis. Built on the lead's established ground truth (run `parallel_benchmark_20260621_055502`, 7 cores @ 3.10 GHz, `paper_exact_fast_growth`, OOM-bound at 26,534 cells). All recreated numbers passed the verifier (all-match). The governing fact is that **only the matched-cell speedup ratio is hardware-portable; every absolute axis is not.**

Live post: `_posts/2026-04-16-simucell3d-adaptive-openmp-scheduling.md`
Figures: `images/blog/simucell3d/` (figure-1 … figure-7, each `.svg` + `.png`)
Datapoints: `/Users/nilesh-patil/.claude/jobs/0da77b3e/tmp/sc3d/datapoints/`

---

## 1. Per-figure update feasibility

| Fig | What it plots | Adaptive line from new run? | v1 / Static line from new run? | New-run source | New-run number vs live-post number (same trend?) |
|---|---|---|---|---|---|
| **1** | thread-imbalance % vs tissue size | **Yes** | **NO** (only adaptive can move) | `sim_adaptive/performance_diagnostics.csv` → `thread_imbalance_pct` | Adaptive cells>100 mean 12.96 / med 12.8 / **max 15.7**; live Adaptive 10.6 / 12.2 / **14.4**. Static line live-only (16.4/16.1/21.2). Same regime, slightly higher max. |
| **2** | workload CoV vs tissue size, 0.4/0.6 thresholds | **Yes** | **NO** (only adaptive can move) | same file → `cov` | Adaptive cells>100 mean 0.130 / med 0.128 / **max 0.157**; live 0.106 / 0.122 / **0.144**. Never crosses 0.4/0.6 in either run. Same trend. |
| **3** | scheduler phase vs iteration/cells | **Yes (full)** | n/a (adaptive-only by design) | same file → `phase` | INIT 40 samples (to iter 4000, ≤9 cells) → HOMEOSTASIS 179 (iter 4100→21900, 10→**26,534**); GROWTH never. Live: INIT→iter 4100→HOMEO to 19,958, GROWTH never. **Reproduces exactly, extends reach.** |
| **4** | cells vs wall-clock (log2-log2) | **Yes** | **Yes** (concurrent) | adaptive `wall_epoch`; v1 = `logs/v1.log` cells ⋈ `sim_v1/simulation_statistics.csv` time | adaptive **26,534 @ 61.79 h** / v1 **12,851 @ 61.80 h**; cells ∝ t^**0.875** (R² 0.996). Live: 19,958/9,693, t^**0.88**. Same exponent. **But absolute time axis ≠ cross-HW comparable.** |
| **4b** | log2 cells vs LINEAR wall-hours (doubling) | **Yes** | **Yes** | adaptive `wall_epoch` rung crossings; v1 join | adaptive rungs 1.60 / 3.90 / 8.03 / 18.69 h (each ~2x prior); v1 3.88 / 8.54 / 18.82 h. Live: ~3 h → ~15 h. Same shape. **Most hardware-sensitive axis — keep single-run.** |
| **5** | iterations/sec vs cell count (log-log) | **Yes** | **Yes** | adaptive IPS = 1000/`total_iteration_ms`; v1 IPS = d(iter)/d(`computation_time` s) | adaptive final 0.00816 @ 26,534; v1 final 0.00792 @ 12,851. α: adaptive **1.092** (R² 0.994), v1 **1.151** (R² **0.986**). Live: adaptive 1.133/0.999, v1 1.132/**0.835**. Same scaling; **new-run v1 fit is far cleaner.** |
| **6** | matched-cell speedup by band (ratio) | **Yes** | **Yes (ratio is the point)** | bandwise median(adaptive IPS)/median(v1 IPS) | 50-99→2.31, 100-249→2.39, 250-499→2.33, 500-999→2.30, 4000-7999→**2.48 peak**, 8000-15999→2.37. Live: 2.0-2.75x with a 250-499 **dip (1.50x)**. **New run is smooth 2.30-2.48x — dip does NOT replicate.** 16000-31999 has no concurrent v1 (v1 caps 12,851). **THE cross-HW figure.** |
| **7** | phase-time share, Static vs Adaptive | **Yes** | **NO** (only adaptive can move) | `performance_diagnostics.csv` per-phase `*_ms` | Adaptive contact **57.3** / polar+internal **29.9** / mesh 6.1 / time-int 6.7. Live Adaptive 54/33/5/8; live Static 82/14/3/2. Same shift; contact share a touch higher. Static line live-only. |

**Adaptive-only flag (v1/Static line cannot come from the new run): figures 1, 2, 7.** Genuine published v1.0 in this run emits no `performance_diagnostics.csv` (no CoV, no `thread_imbalance_pct`, no per-phase `*_ms`). Their Static baseline must stay on the live 8-core run (or the separately instrumented static build, best-matched sibling `20260127_085255`).

---

## 2. Three narrative-update strategies

### (A) REPLACE the hero — re-baseline everything to 7-core @ 3.1 GHz / 26.5k cells
**Pros:** one coherent dataset; bigger headline (26,534 vs 12,851); a cleaner v1 fig5 fit (R² 0.986); the 2.07x ratio is independently re-confirmed.
**Cons (decisive):** figs 1, 2, 7 LOSE their Static/v1 line entirely — these are the *mechanism* figures, so they degrade to adaptive-only or must splice in a different instrumented run (provenance mismatch). The clean "eight cores, same stopping point" framing dies: the new run is OOM-bound at ~98 GB, not a chosen stop. The 250-499 dip the post uses as an honesty beat vanishes. Param also differs (`paper_exact_fast_growth` vs `128k_fast_growth`). Highest rewrite effort.
**Effort:** High (re-source 5-7 figures, rewrite 6+ captions and the headline).
**Ready-to-use sentences:**
- "I re-ran the whole comparison on a different box, seven cores at 3.10 GHz, and pushed it until memory gave out at 26,534 cells against v1's 12,851. Same lever, same factor: 2.07x the tissue for the same wall-clock."
- "One honest difference from the earlier run: this one did not stop cleanly. It was killed at about 98 GB of RAM, so 26,534 is a memory-bound ceiling, not a chosen stopping point."
- "Because published v1.0 emits no per-thread diagnostics, the imbalance, CoV, and phase-time plots here show only the adaptive side; their Static baseline comes from a separately instrumented build."

### (B) OVERLAY / REPLICATE — keep the 8-core hero, add the new run as independent reproduction that also extends reach to 26.5k  *(RECOMMENDED)*
**Pros:** scientifically the most honest framing. The ratio (2.06x → 2.065x) reproduces across different silicon and at 2.6x the cell count; reframes "one big run" as **n = 2 independent runs**. Fig 6 becomes the clean cross-hardware centerpiece (ratio cancels clock and cores). All mechanism figures (1, 2, 7) stay intact on the live run. Matches the post's stated "matched-cell ratio over raw endpoints" preference.
**Cons:** must NOT co-plot absolute axes (4, 4b, 5) — those need small-multiples or a cells-only "extend-the-curve" framing, capped at ~3 series to avoid clutter. The 250-499 dip does not replicate, so an honest overlay shows it in run A but not run B (turn this into an honesty beat).
**Effort:** Medium (1 new slopegraph near TL;DR + upgrade fig 6 to paired two-run bands + a cells-only extension panel; existing figures and numbers untouched).
**Ready-to-use sentences:**
- "A second run, on different silicon and carried 2.6x further in cell count, lands on the same slope. Seven cores at 3.10 GHz grew the tissue to 26,534 cells against v1's 12,851, a 2.07x ratio sitting right on top of the 2.06x from the eight-core run."
- "The honest cross-machine statement is about the ratio, not the raw clock. Wall-clock and iterations per second move the instant the hardware changes, but the matched-cell speedup cancels the clock and the core count, and that is the number that reproduced."
- "I count this as two independent runs, not one big one. The 250-499 dip from the first run did not return in the second, so I no longer lean on it; the 2x envelope did, and that is the claim I trust."

### (C) APPEND a scoped "scaling to 26.5k / reproduced on different hardware" section + 1-2 new figures, existing figures untouched
**Pros:** lowest effort, lowest risk; every existing figure and number stays as-is; the new run lives in one clearly-scoped section with its own provenance stamp and OOM/HW caveats in one place. Good "ship it today" option.
**Cons:** the replication signal is siloed rather than reinforcing each figure; fig 6 is not upgraded into the cross-hardware proof it could be.
**Effort:** Low (one new section, 1-2 figures: a slopegraph + an extended fig-6 band chart).
**Ready-to-use sentences:**
- "After this post first went up I ran it once more on a different machine, partly to see whether the 2x held and partly to push the cell count higher. It did both."
- "On seven cores at 3.10 GHz the adaptive fork reached 26,534 cells before the run was OOM-killed at about 98 GB; the concurrent v1 baseline sat at 12,851, the same 2.07x advantage measured on entirely different hardware."
- "Only the ratio travels between the two machines. The absolute curves above stay on the original eight-core run; this section adds the one quantity, matched-cell speedup, that a clock change cannot move."

**Recommendation: (B), with (C) as the low-effort fallback.** The matched-cell ratio is the only hardware-portable quantity and it reproduced almost exactly (2.06x → 2.065x), so framing the new run as independent replication that also extends reach to 26.5k is both the most honest and the highest-value use. A full hero replacement (A) is rejected because it strips the Static/v1 line out of the three mechanism figures (1, 2, 7) and trades a clean stopping-point story for a memory-bound ceiling.

---

## 3. Recreated-datapoint index (`/Users/nilesh-patil/.claude/jobs/0da77b3e/tmp/sc3d/datapoints/`)

- **fig1_imbalance.csv** (220 lines; `iteration,cells,thread_imbalance_pct`) — adaptive only. all-rows mean 10.76 / med 12.2 / max 15.7; cells>100 (n=128) mean 12.96 / med 12.8 / max 15.7.
- **fig2_cov.csv** (220 lines; `iteration,cells,cov`) — adaptive only. all-rows mean 0.108 / med 0.122 / max 0.157; cells>100 mean 0.130 / med 0.128 / max 0.157. Never ≥ 0.4 or 0.6.
- **fig3_phase.csv** (220 lines; `iteration,cells,phase`) — INITIALIZATION 40 → HOMEOSTASIS 179; transition iter 4000(9 cells)→4100(10 cells); final iter 21900 / 26,534 cells; GROWTH absent.
- **fig4_wallclock.csv** (623 lines; `mode,iteration,cells,wall_hours`) — adaptive 219 rows to 26,534 @ 61.79 h; v1 402 rows to 12,851 @ 61.80 h (matched wall-clock). Adaptive cells ∝ t^0.875, R² 0.996.
- **fig4b_doubling.csv** (`mode,rung_from,rung_to,hours_from,hours_to,delta_hours`) — adaptive rungs 1.60 / 3.90 / 8.03 / 18.69 h; v1 rungs 3.88 / 8.54 / 18.82 h.
- **fig5_ips.csv** (567 lines; `mode,iteration,cells,iter_per_sec`) — adaptive 219 rows (IPS = 1000/total_iteration_ms), v1 347 rows (IPS = d(iter)/d(computation_time s)). adaptive α 1.092 (R² 0.994); v1 α 1.151 (R² 0.986).
- **fig6_speedup_bands.csv** (`cell_band,adaptive_median_ips,n_adaptive,v1_median_ips,n_v1,speedup`) — 10 matched bands 1.11→2.48x; peak 4000-7999 = 2.48; no 250-499 dip; 16000-31999 has no v1 (caps 12,851).
- **fig7_phase_shares.csv** (`phase,mean_ms,share_pct_renorm,share_pct_vs_total`) — adaptive only, cells>100 (n=128): contact 57.32% (9810 ms) / polar+internal 29.85% (5109 ms) / time-int 6.68% (1143 ms) / mesh 6.14% (1051 ms). total = sum of 4 (overhead ~0).
- **physics_check.csv** (404 lines; `iteration,adaptive_cells,v1_cells,adaptive_mean_pressure,v1_mean_pressure,rel_dev_pct`) — 403 matched iterations; median pressure dev 2.33% (mean 4.96%, p90 13.5%, max 40.0% at the high-iteration tail).
- Generator: **`recreate_datapoints.py`** (pure stdlib, Python 3.14, inline least-squares; no pandas).

---

## 4. Autoresearch-backed chart improvements (per figure)

**Cross-cutting (apply to all 7):**
- **Themeability:** inline the SVGs and set structural ink (axes/ticks/grid/text) to `currentColor`, exactly like the post's concept diagrams; drop the opaque white/cream `<rect>` background. `currentColor`/CSS only reaches an *inlined* SVG, not `<img src>`; the site's 3-mode toggle is custom JS, not `prefers-color-scheme` ([Cassidy James](https://cassidyjames.com/blog/prefers-color-scheme-svg-light-dark/), [ctrl.blog](https://www.ctrl.blog/entry/svg-embed-dark-mode.html)). 0 of 7 figures currently do this.
- **Palette:** one fixed Okabe-Ito mode mapping reused everywhere — Adaptive = Blue `#0072B2` solid, Static = Vermillion `#D55E00` dashed (CVD- and grayscale-safe; color + line-style double-encodes) ([Wong 2011 / sci-draw](https://sci-draw.com/blog/colorblind-safe-palettes-okabe-ito-reference)). fig 7 currently uses the red-green Tableau pair `#d62728`/`#2ca02c` — unsafe.
- **Direct-label line ends, drop legends** (figs 1, 2, 4, 4b, 5) ([Depict Data Studio](https://depictdatastudio.com/directly-labeling-line-graphs/), [Practical Reporting](https://www.practicalreporting.com/blog/2024/9/17/avoid-legends-footnotes-and-other-forms-of-indirect-labeling-in-your-charts-whenever-possible)).
- **Annotate the takeaway on-plot** (slope+R², "2x cells" bracket, "0.4 never crossed", "GROWTH never fired") ([data.europa](https://data.europa.eu/apps/data-visualisation-guide/chart-junk-and-data-ink-origins)).
- **Provenance stamp per figure** (`CPU · cores · GHz · param · run-date · single node`), stamped per series when runs are mixed ([SPEC full-disclosure](https://www.spec.org/power/docs/specpower_ssj2008-run_reporting_rules/), [ACM REP'25](https://dl.acm.org/doi/10.1145/3736731.3746150)). Never put a per-core/per-GHz "normalized" number on a chart ([arXiv:2305.07345](https://arxiv.org/abs/2305.07345)).

**Per figure:**
- **fig 1:** redundant — it is fig 2 × 100 (its own caption admits it). Collapse into fig 2, or keep as a small inset; add an on-plot "proxy: workload CoV × 100, not measured barrier wait" note ([data.europa data-ink](https://data.europa.eu/apps/data-visualisation-guide/chart-junk-and-data-ink-origins)).
- **fig 2:** direct-label the 0.4/0.6 reference lines at the right edge; lightly shade the unreached band above 0.4 with "high-CoV branch never exercised on this run" ([HPC Wiki reference lines](https://hpc-wiki.info/hpc/Scaling)).
- **fig 3:** redraw as a Gantt/swimlane phase-band timeline; render GROWTH as a greyed *empty* lane so "never fired" is visible negative space ([Gantt vs swimlane](https://traqplan.com/gantt-chart-vs-swimlane-diagram-which-is-better-for-your-project/)). Extend HOMEOSTASIS label to 26,534 if re-sourced.
- **fig 4:** print exponent ± SE with R²; add the hedge that a high OLS log-log R² does not prove a power law ([Pomona lab](http://www.physics.pomona.edu/sixideas/old/labs/LRM/LR05.pdf), [Shalizi](https://arxiv.org/pdf/nlin/0307015)); draw a "~2x cells" bracket; for the cross-run version use cells-only "extend-the-curve" shading (vertical rule at the ~20k live stop, run-B as open markers continuing the fitted line) with a distinct **OOM marker** on the 26,534 endpoint ([extrapolation caution PMC4619888](https://pmc.ncbi.nlm.nih.gov/articles/PMC4619888/)). Add a compact **slopegraph** companion near the TL;DR (v1→adaptive endpoints, one line per run, log value axis) ([Tufte slopegraphs](https://www.edwardtufte.com/notebook/slopegraphs-for-comparing-gradients-slopegraph-theory-and-practice/)).
- **fig 4b:** keep single-run (linear wall-hours is the most hardware-sensitive axis; a faster clock fakes a doubling-time change); add per-doubling bracket annotations ("~3 h" → "~15 h") ([leancrew](https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/)). If a second curve is wanted, relabel x to core-GHz-hours.
- **fig 5:** annotate α with R²; draw the weaker v1 fit lighter; extend-the-curve shading past v1 termination. The new run actually offers a **cleaner v1 fit** (R² 0.986 vs the live 0.835, because v1 IPS comes from `computation_time` deltas not monitor cleaned data) — worth using even under strategy C ([Pomona lab](http://www.physics.pomona.edu/sixideas/old/labs/LRM/LR05.pdf)).
- **fig 6 (the cross-HW figure):** bold 1.0x baseline + dashed 2.0x reference line; direct-label each bar; **overlay both runs as paired markers per band** (SuperPlots run-coding), aggregate with the **geometric mean** of per-band ratios; add the new run's extended high-cell bands; keep the 10-24/250-499 omission notes ([HPC Wiki](https://hpc-wiki.info/hpc/Scaling), [Fiveable geo-mean](https://fiveable.me/introduction-computer-architecture/unit-8/performance-metrics-benchmarking/study-guide/ofBFIiAG6IRRPXrE), [Datawrapper overlay](https://www.datawrapper.de/blog/what-to-consider-when-creating-small-multiple-line-charts), [SuperPlots PMC7265319](https://pmc.ncbi.nlm.nih.gov/articles/PMC7265319/)).
- **fig 7:** the 100%-stacked form hides that adaptive's *absolute* iteration time shrank (the confusion the prose has to undo). Prefer a **Marimekko** (bar width = absolute mean iteration time) or **grouped absolute-ms bars** or a **dumbbell/slopegraph** of the 82→54 / 14→33 shift; minimum fix = recolor off red-green + direct-label segments + print each bar's absolute ms ([CleanChart stacked-vs-grouped](https://www.cleanchart.app/blog/stacked-vs-grouped-bar-charts), [Marimekko catalogue](https://datavizcatalogue.com/methods/marimekko_chart.html), [Nightingale dumbbell](https://medium.com/nightingale/the-dumbbell-plot-a-how-to-guide-5dafd1d67581)).

---

## 5. Caveats to disclose in prose

1. **OOM, not a clean stop.** The adaptive run was OOM-killed at ~98 GB RAM after iter 21900 / 26,534 cells. This is a *memory-bound ceiling*, not the "same stopping point, eight cores" framing the live run uses. Any figure showing the 26,534 endpoint must mark it as OOM-terminated, not a chosen stop.
2. **v1 endpoint is the concurrent 12,851.** That is v1 at iter 20100 while *sharing* the 7-core machine with adaptive. The v1 **solo tail** (v1 alone on freed cores after adaptive died, to ~15,654 cells at iter ~20561) is archived in `sim_v1/_solo_tail/` and **excluded** from the headline ratio — counting it would be an unequal-resources fairness violation ([arXiv:2305.07345](https://arxiv.org/abs/2305.07345)).
3. **Genuine v1.0 emits no diagnostics.** No CoV, no `thread_imbalance_pct`, no per-phase `*_ms` anywhere in the new run, so the Static/v1 line in **figs 1, 2, 7 cannot come from this run.** They stay on the live 8-core run (or the separately instrumented static build, sibling `20260127_085255`); say so.
4. **Hardware + cores + param all differ** (8c @ 2.80 GHz / `128k_fast_growth` vs 7c @ 3.10 GHz / `paper_exact_fast_growth`). Absolute wall-clock and IPS axes are **not** cross-comparable; only the dimensionless matched-cell speedup is. Never co-plot absolute axes across the two runs; never normalize by clock/cores on-chart.
5. **The 250-499 dip did not replicate.** The new run is a smooth 2.30-2.48x for cells ≥ 50; the live post's 1.50x dip is run A only. If framed as replication, say the dip did not return.
6. **fig 6 smallest bands are quantization-limited** (10-24 = 1.11x, 25-49 = 1.81x): v1 IPS is derived at 1-second `computation_time` resolution, so early v1 IPS collapses to coarse values. Treat the two smallest bands as unreliable; the live post omitted 10-24.
7. **Physics is preserved:** median pressure deviation 2.33% over 403 matched iterations (live: 2.93% / 389). The up-to-40% tail is high-iteration FP non-determinism across thread schedules; the median is the robust headline.
8. **The contact-detection rewrite is still bundled** with the scheduler change (the live post's standing caveat). The new run does not isolate it either; the Morton-off ablation the post flags as missing is still missing.

---

## 6. Open questions for the human

- **Science:** Is the OOM ceiling itself a result worth presenting (memory scaling / ~98 GB at 26.5k cells), or only a caveat? Worth re-running with the instrumented static build to recover a v1 line for figs 1/2/7, or accept adaptive-only there? Should the missing Morton-off ablation block adding a second run, or ship without it (as the live post already does)?
- **Math / stats:** For fig 1/fig 2 captions, which convention — all-219-rows (mean < median, pulled down by ~40 INITIALIZATION zero rows) or the cells>100 regime? The live post's framing is ambiguous; pick one and apply consistently. Geometric vs arithmetic mean for any single fig-6 headline number (research says geometric for ratios)?
- **Writing:** Keep the "same stopping point, eight cores" hero (strategy B/C) or re-baseline to the 26,534 headline (strategy A)? Does the bigger number justify losing the clean-stop story and the three mechanism figures' Static line?
- **Style:** Frame the new run explicitly as **n = 2 independent runs** (SuperPlots discipline) and add a slopegraph near the TL;DR — yes/no?
- **Design:** How far on the figure overhaul — minimum (transparent bg + neutral recolor) vs full inline-SVG + `currentColor` + Okabe-Ito across all 7? Replace fig 7's 100%-stacked bar with a Marimekko/dumbbell, or just recolor and label it?
- **Engineering:** Where do the new figure generators + companion data live? (The site figviz pipeline is NOT committed to the site repo; `recreate_datapoints.py` is pure-stdlib and reproducible.) Inline-SVG figures need a generation path that emits `currentColor` ink rather than matplotlib `<img>` exports.
