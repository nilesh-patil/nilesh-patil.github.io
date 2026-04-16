# C3 - Chart-type best practices, mapped to the 7 SimuCell3D figures

Research date: 2026-06-29. Scope: modern viz best practices for the *specific* chart
types in the post, plus accessibility/themeability. Figure-specific, not generic.

## Current-state audit (measured before recommending)

All 7 measured figures are **matplotlib SVG exports embedded via `<img src=...>`**, not
hand-authored inline SVG. Concrete findings from the files themselves:

- **0 of 7 use `currentColor`.** Every color (axes, ticks, text `#6b7280`, grid
  `#e7e5df`, series colors) is hard-coded. The hand-authored concept diagrams in the
  same post *do* use `currentColor` and recolor with the theme; the measured figures
  do not, so they are out of step with the house style.
- **Backgrounds are opaque and inconsistent:** figures 1, 2, 3, 5, 7 paint `#ffffff`
  (white); figures 4, 4b, 6 paint `#fbfaf7` (cream/sepia). On the dark theme these are
  bright boxes; the two background colors also clash with each other.
- **figure-7 uses the matplotlib Tableau default palette** (`#1f77b4` blue, `#2ca02c`
  green, `#d62728` red, `#ff7f0e` orange). Red+green in the same chart is the classic
  red-green-CVD-unsafe pairing. figure-3 uses orange `#ff7f0e`.
- **Legends present** (Static/Adaptive/v1 text) on figs 1, 2, 5, 7 -> indirect labeling.

This audit drives cross-cutting findings A-F below before the per-figure notes.

---

## Cross-cutting recommendations (apply to most/all figures)

### A. Make the figures themeable the same way the concept diagrams already are
The house style is inline SVG that recolors via `currentColor` + CSS variables across
light/dark/sepia. The measured figures break that. Two facts matter:

1. `currentColor` and page CSS only reach an SVG when it is **inlined into the DOM**, not
   when it is loaded through `<img src>` ([Cassidy James](https://cassidyjames.com/blog/prefers-color-scheme-svg-light-dark/),
   [ctrl.blog](https://www.ctrl.blog/entry/svg-embed-dark-mode.html)).
2. This site's theme is a **custom JS 3-mode toggle** (localStorage light/dark/sepia),
   *not* OS `prefers-color-scheme`. So the usual `@media (prefers-color-scheme: dark)`
   block inside an `<img>`-SVG would not follow the site toggle at all, and it is buggy
   in Safari/Chromium anyway ([MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme),
   [ctrl.blog](https://www.ctrl.blog/entry/svg-embed-dark-mode.html)).

Recommendation, in priority order:
- **Minimum:** in the matplotlib export, drop the opaque background rect (set
  `savefig.facecolor='none'`, `axes.facecolor='none'`) so the themed post background
  shows through, and switch axis/tick/label/grid/spine colors to a single neutral that
  reads on all three themes. This alone fixes the "white box on dark theme" problem.
- **On-brand:** inline the measured SVGs into the post (or convert them to hand-tuned
  inline SVG) and set structural ink (axes, ticks, gridlines, text) to `currentColor` /
  theme CSS vars, exactly like the cell-mesh diagram. Series colors stay fixed (Okabe-Ito,
  below) because data identity should not change with theme. Keep `role="img"` +
  `<title>`/`<desc>` for a11y ([accessible-SVG guidance](https://cassidyjames.com/blog/prefers-color-scheme-svg-light-dark/)).

### B. Replace the Tableau defaults with one consistent, colorblind-safe mode mapping
Use the **Okabe-Ito** palette (the Nature Methods / Wong-2011 categorical standard, CVD-
and grayscale-safe) instead of matplotlib's defaults
([sci-draw](https://sci-draw.com/blog/colorblind-safe-palettes-okabe-ito-reference),
[conceptviz hex reference](https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference)).
Hex: Orange `#E69F00`, Sky Blue `#56B4E9`, Bluish Green `#009E73`, Yellow `#F0E442`,
Blue `#0072B2`, Vermillion `#D55E00`, Reddish Purple `#CC79A7`, Black `#000000`.

Pick **one fixed mapping for the two modes and reuse it in every figure** (1, 2, 4, 4b,
5, 6, 7): e.g. **Adaptive = Blue `#0072B2`** (the hero series), **Static/v1 = Vermillion
`#D55E00`**. Blue/vermillion is a high-luminance-contrast, CVD-safe pair. A consistent
mapping turns 7 separate charts into one visual argument. **Pair color with a second
channel** (solid line = Adaptive, dashed = Static) so the distinction survives grayscale
and CVD; color-plus-shape is safer than color alone ([sci-draw](https://sci-draw.com/blog/colorblind-safe-palettes-okabe-ito-reference)).

### C. Direct-label series; kill the legends (figs 1, 2, 4, 4b, 5)
Place a color-matched label at the **end of each line** rather than a separate legend, so
the reader stops zig-zagging between key and curve. Color the label the same hue (or one
shade darker) as its line, and add the final value at the line end so the reader gets
scale without flicking back to the axis
([Depict Data Studio](https://depictdatastudio.com/directly-labeling-line-graphs/),
[Practical Reporting](https://www.practicalreporting.com/blog/2024/9/17/avoid-legends-footnotes-and-other-forms-of-indirect-labeling-in-your-charts-whenever-possible),
[PolicyViz](https://policyviz.com/2024/01/11/graph-labeling-strategies/)).

### D. Annotate the takeaway *inside* the figure, not only in the caption
Tufte's point: graphics should draw attention to the substance of the data. Put the one
load-bearing number on the plot - the slope/exponent with R^2, the "2x cells" bracket,
"neither curve crosses 0.4", "GROWTH never fired" - so the figure carries its own thesis,
and strip non-data ink (heavy gridlines, redundant labels)
([data.europa chart-junk](https://data.europa.eu/apps/data-visualisation-guide/chart-junk-and-data-ink-origins),
[Dev3lop](https://dev3lop.com/chart-junk-removal-maximizing-data-ink-ratio/)).

### E. Collapse the figure-1 / figure-2 redundancy
figure-2's own caption says it is "the same imbalance signal as figure 1, now in raw CoV":
figure 1 is literally figure 2 multiplied by 100. That is redundant data-ink across two
full figures. Options: (1) keep only figure 2 (raw CoV is the quantity the scheduler
actually thresholds on) and fold the percent framing into its caption; or (2) keep figure
1 as a small inset of figure 2. Carrying both full-size is the kind of redundancy the
data-ink principle warns against ([data.europa](https://data.europa.eu/apps/data-visualisation-guide/chart-junk-and-data-ink-origins)).

### F. Unify the figure frame
One background treatment (transparent), one type family (ideally the site font, not
DejaVu Sans), one aspect ratio family, and the consistent mode colors from B. Right now
backgrounds, fonts, and color logic differ figure to figure.

---

## Per-figure recommendations

### figure-1 - thread imbalance % vs tissue size (log x), Static vs Adaptive
Two-series line chart over a log size axis.
- Apply B (Adaptive blue solid, Static vermillion dashed), C (end-labels, drop legend),
  A (transparent bg + currentColor ink).
- The y-column is a CoV-scaled proxy, not a direct barrier-wait measurement - the caption
  is honest about that; keep that hedge and consider a one-line on-plot note ("proxy:
  workload CoV x 100, not measured barrier wait") so the proxy caveat travels with the
  image (D).
- Consider merging into figure-2 per E. If kept separate, annotate the means on-plot
  (Static ~16%, Adaptive ~11%) rather than only in the caption.
- New-run note: in `20260621_055502` the imbalance max is 15.7% (not 21.2%); any redraw
  from the new run changes these numbers.

### figure-2 - workload CoV vs tissue size, with dashed 0.4 / 0.6 threshold lines
Two-series line chart plus two horizontal reference lines.
- The dashed threshold lines are a textbook good use of **reference lines** (HPC convention
  for ideal/threshold markers, [HPC Wiki Scaling](https://hpc-wiki.info/hpc/Scaling),
  [water-programming scaling](https://waterprogramming.wpcomstaging.com/2021/06/07/scaling-experiments-how-to-measure-the-performance-of-parallel-code-on-hpc-systems/)).
  **Label them directly at the right edge** ("0.4 -> finer chunks", "0.6 -> finest")
  instead of explaining them only in the caption.
- The real story is that *neither curve reaches 0.4*. Make that visual: lightly shade the
  band above 0.4 as the "unreached, high-CoV regime" and add an on-plot note "high-CoV
  branch never exercised on this run" (D). This is more honest and more memorable than the
  prose-only version.
- Apply B and C as in figure-1. This is the figure to keep if you collapse per E.

### figure-3 - scheduler phase decisions vs iteration (linear x), cell count (log y)
This is a **categorical state over time**: the right archetype is a **phase-band /
Gantt-style timeline**, not a scatter. Currently a single orange series.
- Redraw as **horizontal phase bands along the iteration axis** (a 1-lane Gantt /
  swimlane): one colored band for INITIALIZATION (to ~iter 4,100), one for HOMEOSTASIS
  (to the end), each direct-labeled inside the band
  ([Gantt vs swimlane](https://traqplan.com/gantt-chart-vs-swimlane-diagram-which-is-better-for-your-project/)).
- Render **GROWTH as a visible empty/greyed lane** so "never fired" is shown as negative
  space, not just stated. This is the single most effective change here - it makes
  "two of three phases exercised" legible at a glance (D).
- Keep the cell-count as a secondary line or right-axis annotation, and mark the
  ~iter 4,100 INIT->HOMEOSTASIS transition with a labeled rule.
- Use Okabe-Ito hues for the two live phases; reserve a neutral grey for the dead GROWTH
  lane. New-run note: HOMEOSTASIS now runs to 26,534 cells / iter ~21,900.

### figure-4 - cell count vs wall-clock, log2-log2 (power-law, cells ~ t^0.88)
Power-law/scaling curve on log2-log2 axes (every gridline = one doubling: good).
- **Disclose the fit properly:** show the fitted line over the data and print the exponent
  *with R^2 and an uncertainty*, e.g. "cells ~ t^0.88, R^2 = ...". Best practice is to
  report the slope as exponent +/- standard error to 2 decimals and the R^2 alongside
  ([Pomona power-law/log-log lab](http://www.physics.pomona.edu/sixideas/old/labs/LRM/LR05.pdf)).
- **Add the honesty hedge that high R^2 on an OLS log-log fit does not by itself prove a
  power law** - a concave curve can still score R^2 ~0.96. Visual linearity + residuals
  matter ([Shalizi, Complex Systems Science overview](https://arxiv.org/pdf/nlin/0307015),
  [log-log plotting notes](https://www.spsanderson.com/steveondata/posts/2023-10-27/index.html)).
  Since the post already values honest caveats, a one-line on-plot note suffices.
- Draw a faint "~2x cells, same budget" bracket between the two endpoints (D); apply B/C.
- New-run note: endpoints become 26,534 (adaptive) vs 12,851 (v1), and the adaptive run
  was **OOM-killed at ~98 GB, not a clean matched stop** - the caption must say "memory-
  bound stop," and the bracket should compare the concurrent wall-clock-matched point,
  not the unfair v1 solo tail.

### figure-4b - log2 cells vs LINEAR wall hours (doubling-time chart)
This is the canonical **doubling-time semilog**: log-2 y, linear x, gridlines one doubling
apart so the doubling interval is read directly off the chart
([leancrew, exponential growth on base-2 log scales](https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/),
[Semi-log plot overview](https://grokipedia.com/page/Semi-log_plot)). Right chart type; reinforce it:
- **Annotate each doubling as a bracket between two gridlines** with its measured time
  (e.g. "~3 h" at 2k-4k, "~15 h" at 8k-16k) drawn on the plot - this is the recommended
  way to make doubling time legible and it is exactly the post's point that each doubling
  costs ~2x the previous ([leancrew](https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/)).
- Direct-label both curves (C); a straight line would mean constant doubling time, so the
  upward bend *is* the message - call that out on-plot (D).

### figure-5 - iterations/sec vs cell count, log-log (throughput scaling)
Classic log-log throughput curve, two series.
- Apply B/C/A. Annotate the scaling exponent (time/iter ~ N^alpha) **with its R^2** on-plot
  ([Pomona lab](http://www.physics.pomona.edu/sixideas/old/labs/LRM/LR05.pdf)).
- **Show the asymmetry in fit quality honestly:** adaptive uses per-iteration diagnostics
  (R^2 ~0.999) while v1 is monitor-derived (R^2 ~0.835). Draw the weaker v1 fit lighter /
  with a visible confidence band and label its R^2 so readers do not over-trust the
  exponent comparison - the post already flags this in prose; put it on the figure (D).
- Power-law-fit hygiene: fit on **log-spaced points**, drop the first few non-asymptotic
  points, and treat the matched-cell figure-6 as the primary evidence
  ([Shalizi overview](https://arxiv.org/pdf/nlin/0307015)).

### figure-6 - matched-cell speedup by cell band (bar chart, ~2-2.75x, dip at 250-499)
Single-series bar chart of a ratio - the most defensible summary in the post.
- **Add reference lines** the HPC way: a bold **1.0x baseline** (no speedup) and a dashed
  **2.0x** marker (the headline) so every bar is read against a meaningful datum
  ([HPC Wiki Scaling](https://hpc-wiki.info/hpc/Scaling),
  [EPCC analysing performance](https://epcced.github.io/understanding-package-performance/04-analysis/index.html)).
- One hue (Okabe-Ito blue), **direct-label each bar with its value**, and **highlight the
  250-499 dip** in a contrasting/greyed tone with a short annotation so the one honest
  weak spot is shown, not hidden (D). Keep the 10-24 band omitted (already done; note why).
- On-plot note that this matched-cell view is the primary claim and figure-5's raw
  endpoints are secondary - mirrors the post's own framing.

### figure-7 - phase-time share, Static vs Adaptive (currently a 100% stacked bar)
This figure has the most design leverage and the current form fights the message.
- **Why 100% stacked is risky here:** with only 2 bars and 4 segments, only the bottom
  segment (contact) shares a zero baseline; the middle segments (polar+internal, time-int)
  float, and humans compare floating segments poorly (Cleveland-McGill graphical
  perception, [CleanChart stacked-vs-grouped](https://www.cleanchart.app/blog/stacked-vs-grouped-bar-charts),
  [AFFiNE, when to avoid stacked](https://affine.pro/blog/when-to-use-a-stacked-bar-chart)).
- **The honesty problem 100%-stacked creates:** normalizing to 100% hides that adaptive's
  *absolute* mean iteration time is shorter. The post has to explain in prose that
  "polarization became second-largest only because contact got faster, not because that
  work grew" - that is exactly the confusion a part-to-whole-without-totals chart induces
  ([CleanChart](https://www.cleanchart.app/blog/stacked-vs-grouped-bar-charts)).
  Better options, in order:
  1. **Marimekko / mosaic** with **bar width = absolute mean iteration time**: Static's bar
     is wide, Adaptive's is narrower, and the contact segment visibly shrinks in *area*, so
     the reader sees the whole bar got smaller and contact's share fell - both at once
     ([Mosaic plot](https://en.wikipedia.org/wiki/Mosaic_plot),
     [Marimekko catalogue](https://datavizcatalogue.com/methods/marimekko_chart.html),
     [Inforiver Marimekko guide](https://inforiver.com/insights/a-guide-to-marimekko-charts/)).
     Caveat: area is judged less accurately than length, so direct-label every segment.
  2. **Grouped bars by phase** (4 phase groups x 2 modes, common baseline) plotted in
     **absolute ms** - the most accurate read of "did each phase's time go up or down"
     ([CleanChart](https://www.cleanchart.app/blog/stacked-vs-grouped-bar-charts)).
  3. **Slopegraph / dumbbell**, one line per phase connecting Static -> Adaptive: this
     foregrounds the *shift* (contact 82->54, polar 14->33) and the rank swap, which is the
     actual narrative ([Domo slope chart](https://www.domo.com/learn/charts/slope-chart),
     [Nightingale dumbbell how-to](https://medium.com/nightingale/the-dumbbell-plot-a-how-to-guide-5dafd1d67581),
     [Domo dumbbell](https://www.domo.com/learn/charts/dumbbell-plot-chart)).
- If the 100% stacked form is kept for simplicity, the **minimum fixes** are: (a) recolor
  off red+green to Okabe-Ito (the current `#d62728`/`#2ca02c` pairing is red-green-unsafe),
  (b) **direct-label each segment with its %** and drop the legend, and (c) print the
  **absolute mean iteration time of each bar** next to it so the "whole bar shrank" fact is
  not lost.

---

## Source list (URLs)

- Power-law / log-log / R^2: <http://www.physics.pomona.edu/sixideas/old/labs/LRM/LR05.pdf>,
  <https://arxiv.org/pdf/nlin/0307015>, <https://www.spsanderson.com/steveondata/posts/2023-10-27/index.html>
- HPC scaling / speedup reference lines: <https://hpc-wiki.info/hpc/Scaling>,
  <https://waterprogramming.wpcomstaging.com/2021/06/07/scaling-experiments-how-to-measure-the-performance-of-parallel-code-on-hpc-systems/>,
  <https://epcced.github.io/understanding-package-performance/04-analysis/index.html>
- Semilog / doubling-time: <https://leancrew.com/all-this/2020/03/exponential-growth-and-log-scales/>,
  <https://grokipedia.com/page/Semi-log_plot>
- Stacked vs grouped vs 100%: <https://www.cleanchart.app/blog/stacked-vs-grouped-bar-charts>,
  <https://affine.pro/blog/when-to-use-a-stacked-bar-chart>
- Marimekko / mosaic: <https://en.wikipedia.org/wiki/Mosaic_plot>,
  <https://datavizcatalogue.com/methods/marimekko_chart.html>,
  <https://inforiver.com/insights/a-guide-to-marimekko-charts/>
- Dumbbell / slopegraph: <https://www.domo.com/learn/charts/dumbbell-plot-chart>,
  <https://medium.com/nightingale/the-dumbbell-plot-a-how-to-guide-5dafd1d67581>,
  <https://www.domo.com/learn/charts/slope-chart>
- Phase timeline / Gantt-swimlane: <https://traqplan.com/gantt-chart-vs-swimlane-diagram-which-is-better-for-your-project/>
- Colorblind-safe palette (Okabe-Ito): <https://sci-draw.com/blog/colorblind-safe-palettes-okabe-ito-reference>,
  <https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference>
- Direct labeling over legends: <https://depictdatastudio.com/directly-labeling-line-graphs/>,
  <https://www.practicalreporting.com/blog/2024/9/17/avoid-legends-footnotes-and-other-forms-of-indirect-labeling-in-your-charts-whenever-possible>,
  <https://policyviz.com/2024/01/11/graph-labeling-strategies/>
- Themeable / dark-mode SVG: <https://cassidyjames.com/blog/prefers-color-scheme-svg-light-dark/>,
  <https://www.ctrl.blog/entry/svg-embed-dark-mode.html>,
  <https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme>
- Chart junk / data-ink / annotate-the-insight: <https://data.europa.eu/apps/data-visualisation-guide/chart-junk-and-data-ink-origins>,
  <https://dev3lop.com/chart-junk-removal-maximizing-data-ink-ratio/>
