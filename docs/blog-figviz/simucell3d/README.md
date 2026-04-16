# SimuCell3D adaptive-OpenMP blog figures

Reproducible figure pipeline for `_posts/2026-04-16-simucell3d-adaptive-openmp-scheduling.md`.
The seven `figure-*.svg` charts under `images/blog/simucell3d/` are generated here.

## What the figures show

Two benchmark runs of the SimuCell3D fork are overlaid:

The two runs are labelled **RUN01** and **RUN02** consistently across the post and
the figures:

| Run | Dir under `<simucell3d>/docs/simulation_results/` | Cores | Adaptive max | v1 max |
|---|---|---|---|---|
| **RUN01** (original, published) | `parallel_benchmark_20260124_091702` | 8 | **19,958** | 9,693 |
| **RUN02** (folds in 25k+) | `parallel_benchmark_20260621_055502` | 7 | **26,534** | 12,851 |

Cell counts are reconciled to each run's **largest** count (so RUN01 is
19,958 everywhere, not the 19,912 monitor endpoint or the 19,614 checkpoint).

Per the locked plan:
- figs 3,4,4b,5,6 co-plot **both** runs (RUN01 solid, RUN02 dashed where they are lines).
- figs 1,2,7 show RUN02's adaptive run plus RUN01's Static/v1 as a
  dashed reference (genuine v1.0 in RUN02 emits no diagnostics, so there is
  no RUN02 Static curve).

## Provenance note: the recovered Static cloud (figs 1 & 2)

The original run's `metadata.json` lists `simulations: [v1, static, adaptive]`, but
only `sim_adaptive` (diagnostics) and `sim_v1` (no diagnostics) survive on disk; the
static-mode diagnostics that produced the published "Static (v1)" CoV / imbalance
clouds were pruned, and **no surviving CSV reproduces them** (committed CoV max
0.212). The blog figviz pipeline that drew them was also discarded.

`extract_static.py` recovers that published Static cloud directly from the committed
`figure-1.svg` / `figure-2.svg`: the adaptive cloud (208 points) has known data
(the 20260124 `performance_diagnostics.csv`), so it calibrates the device<->data
transform (log10 cells on x, linear value on y); inverting it on the 137-point
Static cloud recovers the points. Recovered stats match the published caption to
four decimals (imbalance 16.41/16.10/21.20; CoV 0.1641/0.1607/0.2120). Outputs:
`datapoints/fig1_static_recovered.csv`, `datapoints/fig2_static_recovered.csv`
(committed, because the raw source is gone).

## Regenerate

Needs a matplotlib-capable interpreter (the originals were Matplotlib 3.10.x).

```bash
# from this directory; SIMUCELL3D_RESULTS overrides the run-results root
/opt/anaconda3/bin/python3 extract_static.py     # (re)build the recovered Static clouds
/opt/anaconda3/bin/python3 recreate_datapoints.py  # new-run datapoint CSVs (stdlib only)
/opt/anaconda3/bin/python3 make_figures.py        # render all 7 svg+png into images/blog/simucell3d/
```

`make_figures.py` prints a verification block; the original run reproduces every
committed number (19,958 / 9,693, CoV 0.144, phases 41/167, shares 54/33/8/5,
adaptive alpha 1.133 R2 0.999, v1 alpha 1.132 R2 0.835, physics 2.93% over 389
iters, bands incl. the 250-499 dip and the 150-249 peak of 2.75x).

## Files

- `make_figures.py` - the figure generator (reads both runs + recovered Static clouds).
- `extract_static.py` - Rosetta-Stone recovery of the published Static cloud from SVGs.
- `recreate_datapoints.py` - stdlib-only datapoint CSVs for the new run.
- `datapoints/` - recovered Static clouds + new-run figure datapoints.
- `PROPOSAL.md`, `research/` - the planning notes and chart-improvement research.
- `new_run_recreated_summary.json` - all new-run headline numbers, verified.
