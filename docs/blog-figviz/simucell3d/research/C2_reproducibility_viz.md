# C2 - Visualization idioms for reproducibility & scaling-extension (SimuCell3D figs)

Scope: how to show that a **second independent run** (`20260621_055502`, 7 cores @3.10GHz,
paper_exact_fast_growth) **replicates** the live-post finding (adaptive ~= 2.06x v1) and
**extends its reach** (20k -> 26.5k cells), without overstating it. Focus figures: 4,
4b, 5, 6 and the overall narrative. All recommendations honour the house style
(themeable inline SVG via `currentColor`; direct value labels; no em-dashes; matched-cell
comparison preferred over raw endpoints; honest about caveats).

---

## 0. The one caveat that governs every cross-run figure

The two runs are on **different machines**: live post = 8 cores @ 2.80 GHz; new run =
7 cores @ 3.10 GHz. CPU throughput is not one-dimensional, and core-count and clock both
move the result, so **raw wall-clock and raw iterations/sec are NOT comparable across the
two runs** (Tom's Hardware; isitBOTTLENECKED; arXiv SPEC analysis). The standard fix in
benchmarking is to **compare ratios normalized to a reference, not raw times** (RL4CO
PassMark normalization; IAC reference-machine baseline). Consequence for this post:

- The **reproducible quantity is the matched-cell speedup ratio** (fig 6 and the 2.06x /
  2.065x headline). The ratio cancels the hardware difference, so "both runs land at ~2.06x"
  is a clean replication claim.
- Absolute-position figures with a **time or rate axis** (4, 4b, 5) cannot carry the
  replication claim directly: a faster clock shifts the whole curve. If a second run is
  overlaid there, it must be on a **hardware-robust axis** (cell count), or the hardware
  delta must be annotated, or it stays single-run.

Sources: https://www.tomshardware.com/reviews/cpu-hierarchy,4312.html ,
https://www.isitbottlenecked.com/blog/cpu-core-count-vs-clock-speed ,
https://arxiv.org/pdf/2401.16690 , https://arxiv.org/pdf/2306.17100 (RL4CO PassMark) ,
https://arxiv.org/pdf/1702.04942 (reference-machine normalization).

---

## 1. Idiom catalogue (mapped to figures)

### A. Slopegraph for the headline replication (fig 4 + overall TL;DR)
A slopegraph is the canonical "this case vs that case / before vs after, exactly two
conditions" idiom (Tufte; Forum One; Domo). Here the two conditions are **v1 -> adaptive**,
and the two runs become **two near-parallel lines**:

- Left axis = v1 endpoint cells; right axis = adaptive endpoint cells.
- Line 1 (Run A, live post): 9,693 -> 19,958. Line 2 (Run B, new): 12,851 -> 26,534.
- The two slopes are **near-identical** (2.06x and 2.065x). In a slopegraph, parallel
  slopes ARE the message: same direction, same magnitude of change, replicated. "Lines of
  unusual slope stand out"; here the point is that neither does.
- Tufte rules to apply: direct-label the numeric value at **both** endpoints of each line;
  use thin gray connector lines that do not crash into the numbers; for multiplicative data
  **take logs of the value axis** (otherwise the graphic gets very tall and distorts the
  relative comparison). Spacing of labels proportional to value.
- This is a tiny, high-data-ink panel that proves replication in one glance and is far
  cleaner than overlaying four raw trajectories.

Sources: https://www.edwardtufte.com/notebook/slopegraphs-for-comparing-gradients-slopegraph-theory-and-practice/ ,
https://www.forumone.com/insights/blog/good-data-visualization-practice-slopegraphs/ ,
https://www.domo.com/learn/charts/slope-chart .

### B. SuperPlots two-tier framing: run-level markers on top of per-iteration data (figs 4, 5)
SuperPlots (Lord et al. 2020, JCB) solve exactly this problem in cell biology: don't let the
thousands of low-level points (here: per-iteration samples) drown the **n = number of
independent runs**. Their recipe: plot the raw data, then **overlay a summary marker per
replicate, color-coded by replicate**, and compute the headline statistic on n = replicates,
not n = cells. Map to our figs:

- Keep the per-iteration adaptive/v1 trajectory as the faint background.
- Overlay a **bold endpoint marker per run, color/shape-coded by run** (Run A = filled
  circle, Run B = filled triangle), exactly the SuperPlots "yellow dots / gray triangles /
  blue squares per experiment" encoding.
- State explicitly that the 2.06x headline is computed on **n = 2 runs**, not n = thousands
  of iterations. This is the honest reproducibility framing and matches the house "matched
  comparison" preference.

Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC7265319/ ,
https://rupress.org/jcb/article/219/6/e202001064/151717/ .

### C. "Extend the curve" shaded panel (figs 4, 5)
The forecast/projection idiom: distinguish the **already-observed range** from the **newly
reached range** with a shaded rectangle + a vertical reference line at the original run's
stopping point, using a light, high-transparency fill so the data line stays dominant
(Plotly shapes; DataCamp; matplotlib annotate; Excel shaded-area guides).

- Vertical reference line at ~20k cells (live-post stop). Shade everything to the right
  (20k -> 26.5k) lightly, labelled "new run reaches here". This makes the **scaling-extension**
  visually literal: the trend does not just hold, it carries 33% further.
- Per the extrapolation literature, the strongest version is that the new run's measured
  points **replace what would otherwise be extrapolation**: where run A could only fit a
  line up to 20k, run B supplies real points to 26.5k that fall on the same line. That is
  evidence, not a forecast.

Sources: https://www.datacamp.com/tutorial/plotly-shapes-guide ,
https://matplotlib.org/stable/gallery/text_labels_and_annotations/annotation_demo.html ,
https://pmc.ncbi.nlm.nih.gov/articles/PMC4619888/ (caution on extrapolating past observed data).

### D. Power-law slope as the replicated invariant (figs 4, 5)
On log-log axes a power law is a straight line whose **slope is the exponent**; replication
of a scaling law means the straight line **persists over a wider range** (deep-learning
scaling-law practice; The Math Doctors; Lil'Log). Concretely:

- Fig 4 already states cells ~ t^0.88; fig 5 states time/iter ~ N^1.133. Annotate the slope
  directly on the line (label the exponent on-plot), per power-law best practice.
- The replication claim is: **run B's points lie on the same-slope line, now extended to
  26.5k**. Fit the slope on **log-spaced points** and note the fit range; show run B as open
  markers continuing run A's fitted line. If they hug the line, reproducibility is
  self-evident with no extra rhetoric.
- Honesty hook the literature demands: do not extrapolate the fit far below the smallest
  point, and flag that v1's exponent (R^2 = 0.835) is monitor-derived and weak. The post
  already does this; keep it.

Sources: https://mbrenndoerfer.com/writing/power-laws-deep-learning-neural-network-scaling ,
https://lilianweng.github.io/posts/2026-06-24-scaling-laws/ ,
https://www.themathdoctors.org/logarithmic-graphing/ .

### E. Small multiples vs overlay - the clutter ceiling (figs 4, 5, 6)
Overlaying more than ~3 series destroys legibility; beyond that, use small multiples with
**identical axes, identical chart type, identical size, identical measures** (Forum One;
xdgov Data Design Standards; CleanChart). Our cross-run figs naively have 4 series
(v1/adaptive x 2 runs). Two clean resolutions:

- **Small multiples:** one panel per run, same axes, stacked or side by side. Reader compares
  panel-to-panel; the shape repeats = replication. Best for figs 4 and 5.
- **Collapse to the ratio:** fig 6 collapses each run to a single ratio series, so overlaying
  both runs is only 2 lines, under the clutter ceiling. This is why the ratio figure is the
  right place to overlay runs.

Avoid drawing four raw trajectories on one axis.

Sources: https://www.forumone.com/insights/blog/good-data-visualization-practice-small-multiples/ ,
https://xdgov.github.io/data-design-standards/components/small-multiples ,
https://www.cleanchart.app/blog/scientific-data-visualization .

### F. Accumulation / "the curve smooths, shape unchanged" (figs 4, 4b, 5)
Adding data points refines a curve without changing its basic shape; watching bumpiness
decline as points accumulate is itself evidence of convergence (FlowingData; Claus Wilke
trends chapter). Framing for the post: run B is not a different experiment, it is **more of
the same curve**; show its points as a continuation (open markers, same color family) rather
than a competing series.

Sources: https://flowingdata.com/2018/07/09/how-to-visualize-recurring-patterns/ ,
https://clauswilke.com/dataviz/visualizing-trends.html .

---

## 2. Per-figure recommendations

### Figure 4 (cells vs wall-clock, log2-log2) - PRIMARY replication figure
1. Add a compact **slopegraph inset or companion** (idiom A): v1->adaptive endpoints, one
   line per run, both labelled with their numbers and their ratio (2.06x, 2.065x). This is
   the cleanest possible replication statement and belongs near the TL;DR too.
2. Do NOT overlay run B's trajectory on the **wall-clock x-axis** without a caveat: the new
   run's clock is 3.10 GHz vs 2.80 GHz, so its curve sits left of run A's for hardware
   reasons, not scheduler reasons (idiom 0). If you overlay, annotate the hardware delta and
   keep the comparison qualitative.
3. Better cross-run overlay lives on a **cells-only framing** (idiom C/D): keep run A's
   fitted t^0.88 line, add run B as open markers, shade the 20k->26.5k extension region, mark
   the original stop with a vertical reference line.
4. Mark run B's endpoint with a **distinct "OOM / memory-bound" marker**, not a matched-budget
   stop. The new run was OOM-killed at ~98 GB, 26,534 cells; it is NOT a clean "same stopping
   point". Honest-caveat house style requires the endpoint marker to say so. Update the
   "same stopping point" caption language for the cross-run version.
5. Label the exponent (t^0.88) directly on the line (idiom D).

### Figure 4b (log2 cells vs LINEAR wall hours) - keep single-run
- This figure is the **most hardware-sensitive** of all: the x-axis is wall-clock hours, and
  a faster clock compresses the whole curve horizontally. Overlaying run B here would imply a
  doubling-time change that is really a clock change. Recommendation: **leave 4b single-run**
  (live post run), or if a second curve is wanted, switch the x-axis to a
  hardware-normalized unit (core-GHz-hours) and say so. Doubling-time-grows-2x-per-rung is a
  within-run structural finding; do not turn it into a cross-run claim.

### Figure 5 (iterations/sec vs cell count, log-log) - replication via slope, not height
1. The **height** (absolute IPS) is hardware-dependent and will differ between runs; do not
   present run B's IPS as matching run A's. The **slope** (N^1.133) is the replicated
   invariant. Annotate the exponent on-plot and show run B continuing the same-slope line to
   26.5k (idiom D + C).
2. Apply the **extend-the-curve shading** here too: v1 terminates at ~9.7k/12.9k; adaptive
   carries to ~20k (run A) / 26.5k (run B). Shade the post-v1 region to dramatize "adaptive
   keeps going where v1 stops", and the post-20k region to show run B's extension.
3. If overlaying both runs, use **small multiples** (idiom E), one panel per run, identical
   log-log axes, so the repeated slope reads as replication.

### Figure 6 (matched-cell speedup by band) - THE reproducibility figure
1. This is the **only figure where the cross-run claim is fully clean**, because the ratio
   cancels hardware (idiom 0). Make it the centrepiece of the replication argument.
2. **Overlay both runs as paired markers per band** (grouped bars, or two dots per band
   connected like a mini-slopegraph), color/shape-coded by run (SuperPlots encoding, idiom B).
   The reproducibility message: both runs sit in the same 2.0-2.75x envelope, with the **same
   midrange dip near 250-499 cells appearing in both**. A replicated dip is far more
   convincing than a replicated headline number.
3. Add the new run's **extended bands** (the cell bands above ~20k that only run B reaches) as
   additional bars on the right - this is "extend the reach" inside the matched-comparison
   figure, the most defensible place to show it.
4. Keep the n-omission caveats (10-24 band omitted; one cleaned sample) per run.

### Figure 7 (phase-time share, Static vs Adaptive) - bonus replication small-multiple
- Not in the focus list, but note: this is a literal two-condition comparison and is a
  natural **slopegraph** (contact 82->54, polar+internal 14->33, etc.). With two runs it
  becomes a **small-multiple of two slopegraphs**; if the new run reproduces the
  contact-detection-share collapse and the polarization rise, that is strong corroboration.
  v1 in the new run has no per-phase diagnostics, so this can only be done for the adaptive
  side or qualitatively - flag that limit.

---

## 3. Overall-narrative recommendations

1. **Lead the replication with the ratio, not the raw numbers.** Because the two runs are on
   different hardware, the honest cross-run statement is "the 2x advantage reproduces" (ratio),
   not "the same throughput reproduces". Put the slopegraph (idiom A) up near the TL;DR.
2. **Frame run B as continuation, not a second experiment** (idiom F): same curve, more of it.
   "A second independent run on different hardware lands on the same slope and carries it 33%
   further, to 26.5k cells." Use open markers / shaded extension rather than a fourth color.
3. **Be explicit about the two honesty caveats** (house style): (a) the runs differ in clock
   and core count, so only the ratio is directly comparable; (b) run B's 26,534-cell endpoint
   is memory-bound (OOM at ~98 GB), not a matched-budget stop, and v1's solo tail to ~15.6k is
   excluded as unfair. Both belong in figure captions, not just prose.
4. **Cap overlays at the ratio figure.** Replication shown four-curves-on-one-axis will clutter
   (idiom E); show it via the slopegraph (endpoints), the small multiples (trajectories), and
   the paired-band ratio (fig 6). Three idioms, each under the clutter ceiling.
5. **Statistic on n = runs.** Borrow the SuperPlots discipline: the headline 2.06x is an n = 2
   independent-run result; say so. It reframes "one big run" into "a replicated finding", which
   is the whole point of this task.

---

## 4. Priority build list (what to actually make)

1. **Slopegraph of v1->adaptive endpoints, 2 lines (one per run)** - new small figure near the
   TL;DR. Highest payoff, lowest effort, fully hardware-robust. (idiom A)
2. **Fig 6 upgraded to paired two-run bands** + extended high-cell bands - the clean
   reproducibility centrepiece. (idioms B, C, E)
3. **Fig 4/5 extend-the-curve shading + run-B open markers on the cells axis + on-plot slope
   label + OOM endpoint marker** - the scaling-extension visual. (idioms C, D, F)
4. **Leave 4b single-run** (or relabel x to hardware-normalized units). (idiom 0)
5. Optional: **adaptive-side phase small-multiple** (fig 7) if the new run reproduces the
   contact->polarization shift. (idiom A as small multiple)

---

## Sources (URLs used)
- Small multiples: https://www.forumone.com/insights/blog/good-data-visualization-practice-small-multiples/ , https://xdgov.github.io/data-design-standards/components/small-multiples , https://www.cleanchart.app/blog/scientific-data-visualization , https://flowingdata.com/2018/07/09/how-to-visualize-recurring-patterns/
- Slopegraphs: https://www.edwardtufte.com/notebook/slopegraphs-for-comparing-gradients-slopegraph-theory-and-practice/ , https://www.forumone.com/insights/blog/good-data-visualization-practice-slopegraphs/ , https://www.domo.com/learn/charts/slope-chart , https://inforiver.com/insights/slopegraphs-guide/
- SuperPlots / replication: https://pmc.ncbi.nlm.nih.gov/articles/PMC7265319/ , https://rupress.org/jcb/article/219/6/e202001064/151717/ , https://www.molbiolcell.org/doi/10.1091/mbc.E21-03-0130 (Violin SuperPlots) , https://arxiv.org/pdf/1605.08749 (inline replication)
- Trends / fitting / extrapolation: https://clauswilke.com/dataviz/visualizing-trends.html , https://pmc.ncbi.nlm.nih.gov/articles/PMC4619888/
- Power-law / log-log scaling: https://mbrenndoerfer.com/writing/power-laws-deep-learning-neural-network-scaling , https://lilianweng.github.io/posts/2026-06-24-scaling-laws/ , https://www.themathdoctors.org/logarithmic-graphing/ , https://arxiv.org/pdf/1605.06972
- Annotation / shaded forecast region: https://www.datacamp.com/tutorial/plotly-shapes-guide , https://matplotlib.org/stable/gallery/text_labels_and_annotations/annotation_demo.html
- Cross-hardware benchmark caveats / normalization: https://www.tomshardware.com/reviews/cpu-hierarchy,4312.html , https://www.isitbottlenecked.com/blog/cpu-core-count-vs-clock-speed , https://arxiv.org/pdf/2401.16690 , https://arxiv.org/pdf/2306.17100 , https://arxiv.org/pdf/1702.04942
