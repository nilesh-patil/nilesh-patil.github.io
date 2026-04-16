# line drawings

The seven concept drawings in the SimuCell3D post (`fig-mesh`, `fig-forces`, `fig-divide`, `fig-iteration`, `fig-grid`, `fig-threads`, `fig-scheduler`) are inline SVG in the post itself. This folder holds the script that draws them, so a label can be changed in one place.

```bash
python3 build_figs.py   # needs numpy; writes out/<id>.svg
```

Each output file is one `<svg>` element. Paste it between `<figure>` and `<figcaption>` in the post with every line flush left and no blank lines inside.

Rules the script keeps to: `viewBox` 720 wide (the figure block is 756 px on a laptop), strokes and text in `currentColor` so the drawing follows the light, dark and sepia themes, one accent per drawing through `var(--global-accent, currentColor)`, and `role="img"` with a `<title>` and `<desc>`.

`fig-threads` and the short phases in `fig-iteration` are schematic, not measured. In `fig-iteration` only the width of the contact block (about three quarters of an iteration at 100,000 cells) comes from the benchmark.
