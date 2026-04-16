# bench2 figures

Four figures in `images/blog/simucell3d/` (a fifth with `--memory`), all T1 (the lab's growing-aggregate file, contact model 1) at dt 2.5e-8, 8 threads on one GCP c2-standard-16:

- `bench2-reach`: final cell count per build and run, runs that stopped early in red.
- `bench2-growth`: cells against wall-clock hours, with the three early stops of v1.0 + two patches marked.
- `bench2-sec-per-iter`: seconds per iteration against cells, with the stated N^1.08 and N^1.16 slopes (labelled "about N^1.1 and N^1.2").
- `bench2-matched-cases`: hours for v1.0 + two patches and v2.1.0 on six matched cases.
- `bench2-memory`: resident memory against cells. Not embedded in the post, so only written with `--memory`.

Printed labels are rounded and use the post's build names ("about 130 cells", "about 25k", "2.5 h", "16 to 17 h", "stopped early", "two-day cap"). The CSVs keep the exact values and the artifact's series names: `v1.0` is printed as "v1.0 as tagged" and `v1.0 + crash fixes` as "v1.0 + two patches". Marks are placed from the exact values.

Each figure reads one CSV in `datapoints/` (plus `bench2_setup.csv` for subtitles). Every row has a `source` column. All values were extracted on 2026-11-22 from the artifact "SimuCell3D Growth Reproduction" (version 1795354920-9a8c): the curves from its `b2data` keys `growth`, `spi` and `mem`, the final counts from its `M` and `ROWS` tables. The matched-case hours come from the companion artifact "SimuCell3D v1.0 vs 2.x", table "Speed on matched cases".

Caveats: one VM type only. v1.0 writes no diagnostics, so its timings come from its log. The `growth` and `spi` series are downsampled, so read milestones from `bench2_reach.csv`, not the curves. The artifact writes the v2.1.0 memory rate as "0.13 GB"; the figure labels it GiB because that matches its own data (18.86 GiB at 145,321 cells).

```bash
python3 -I extract_bench2_datapoints.py <saved-artifact.html>   # stdlib only
python3 make_bench2_figures.py                                   # matplotlib 3.10.8
```
